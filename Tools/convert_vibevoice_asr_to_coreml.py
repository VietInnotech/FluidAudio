#!/usr/bin/env python3
"""Convert VibeVoice-ASR to CoreML for on-device Apple inference.

Produces the 4-model pipeline used by FluidAudio's VibeVoiceAsrManager:
  1. vibevoice_acoustic_encoder.mlpackage   — raw audio → acoustic features
  2. vibevoice_semantic_encoder.mlpackage   — raw audio → semantic features
  3. vibevoice_decoder_stateful.mlpackage   — stateful Qwen2.5-7B decoder with fused lmHead
  4. vibevoice_embeddings.bin               — float16 token embedding matrix
  5. vocab.json                             — tokenizer vocabulary
  6. metadata.json                          — model configuration

Usage:
    # Convert all components
    uv run convert_vibevoice_asr_to_coreml.py

    # Convert only specific components
    uv run convert_vibevoice_asr_to_coreml.py --components decoder
    uv run convert_vibevoice_asr_to_coreml.py --components encoders
    uv run convert_vibevoice_asr_to_coreml.py --components embeddings,vocab

    # Custom output directory
    uv run convert_vibevoice_asr_to_coreml.py --output-dir ./build/vibevoice-asr-coreml

Architecture:
  VibeVoiceASRForConditionalGeneration
    ├── acoustic_tokenizer   → vibevoice_acoustic_encoder.mlpackage (encoder only)
    ├── semantic_tokenizer   → vibevoice_semantic_encoder.mlpackage (encoder only)
    ├── audio_projector      → fused into encoder outputs
    ├── llm (Qwen2.5-7B)    → vibevoice_decoder_stateful.mlpackage
    │   ├── embed_tokens     → vibevoice_embeddings.bin (float16)
    │   ├── layers[0..27]    → stateful KV cache decoder
    │   └── norm             → fused into decoder
    └── lm_head              → fused into decoder

Based on FluidInference/mobius conversion scripts.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture constants — auto-detected from config.json
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "microsoft/VibeVoice-ASR"
SAMPLE_RATE = 24000
COMPRESSION_RATIO = 3200
MAX_AUDIO_SECONDS = 3600.0  # 60 minutes


def load_model_config(model_id: str) -> dict:
    """Load and parse architecture config from HuggingFace."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return config


def get_arch_constants(config: dict) -> dict:
    """Extract architecture constants from config.json.

    VibeVoice-ASR uses the keys:
      - 'acoustic_tokenizer_config'   (not 'acoustic_tokenizer')
      - 'semantic_tokenizer_config'   (not 'semantic_tokenizer')
      - 'decoder_config'              (not 'llm_config' / 'text_config')
    """
    # Acoustic tokenizer config — key is 'acoustic_tokenizer_config'
    acoustic_cfg = config.get(
        "acoustic_tokenizer_config",
        config.get("acoustic_tokenizer", {})
    )
    # Semantic tokenizer config — key is 'semantic_tokenizer_config'
    semantic_cfg = config.get(
        "semantic_tokenizer_config",
        config.get("semantic_tokenizer", {})
    )
    # LLM (Qwen2.5-7B) config — key is 'decoder_config'
    llm_cfg = config.get(
        "decoder_config",
        config.get("llm_config", config.get("text_config", {}))
    )

    # The VAE latent dims are stored at the top level in VibeVoice-ASR
    acoustic_vae_dim = config.get("acoustic_vae_dim", acoustic_cfg.get("vae_dim", 64))
    semantic_vae_dim = config.get("semantic_vae_dim", semantic_cfg.get("vae_dim", 128))

    hidden_size = llm_cfg.get("hidden_size", 3584)
    num_q_heads = llm_cfg.get("num_attention_heads", 28)

    return {
        # Acoustic tokenizer
        "acoustic_latent_dim": acoustic_vae_dim,
        "acoustic_strides": acoustic_cfg.get("encoder_ratios", acoustic_cfg.get("encoder_strides", [8, 5, 5, 4, 2, 2])),
        "acoustic_d_model": acoustic_cfg.get("encoder_n_filters", 32),
        "acoustic_channels": acoustic_cfg.get("encoder_n_filters", 32),
        # Semantic tokenizer
        "semantic_latent_dim": semantic_vae_dim,
        "semantic_d_model": semantic_cfg.get("encoder_n_filters", 32),
        "semantic_strides": semantic_cfg.get("encoder_ratios", semantic_cfg.get("encoder_strides", [8, 5, 5, 4, 2, 2])),
        # Audio projector (connector)
        "audio_projector_hidden_size": hidden_size,
        # Text decoder (Qwen2.5-7B)
        "hidden_size": hidden_size,
        "intermediate_size": llm_cfg.get("intermediate_size", 18944),
        "num_layers": llm_cfg.get("num_hidden_layers", 28),
        "num_q_heads": num_q_heads,
        "num_kv_heads": llm_cfg.get("num_key_value_heads", 4),
        "head_dim": hidden_size // num_q_heads,
        "vocab_size": llm_cfg.get("vocab_size", 152064),
        "rope_theta": llm_cfg.get("rope_theta", 1000000.0),
        "rms_norm_eps": llm_cfg.get("rms_norm_eps", 1e-6),
    }


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_all_weights(model_id: str) -> dict:
    """Download and load all safetensors weights, handling multi-shard models."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"  Multi-shard model: {len(shard_files)} shards")

        all_weights = {}
        for shard_file in shard_files:
            shard_path = hf_hub_download(model_id, shard_file)
            shard_weights = load_file(shard_path)
            all_weights.update(shard_weights)
            print(f"    Loaded {shard_file}: {len(shard_weights)} tensors")
        return all_weights

    except Exception:
        st_path = hf_hub_download(model_id, "model.safetensors")
        return load_file(st_path)


# ---------------------------------------------------------------------------
# Fused Stateful Decoder (Qwen2.5-7B)
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE rotation using concatenated-halves layout."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped query attention."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class FusedStatefulQwen2Decoder(nn.Module):
    """Qwen2.5-7B decoder with fused lmHead and stateful KV cache for CoreML export.

    Key difference from Qwen3: NO QK norms (q_norm/k_norm not present in Qwen2.5).
    Uses standard RoPE (not multi-RoPE).
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Linear,
        arch: dict,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.max_seq_len = max_seq_len
        self.num_layers = arch["num_layers"]
        self.num_q_heads = arch["num_q_heads"]
        self.num_kv_heads = arch["num_kv_heads"]
        self.head_dim = arch["head_dim"]
        self.hidden_size = arch["hidden_size"]
        self.gqa_repeat = self.num_q_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Register state buffers (CoreML states must be fp16)
        for i in range(self.num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim, dtype=torch.float16),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - q_len

        cos = position_cos.unsqueeze(1)  # [1, 1, q_len, head_dim]
        sin = position_sin.unsqueeze(1)

        for i in range(self.num_layers):
            layer = self.layers[i]
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")

            # Pre-attention LayerNorm
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            # Q, K, V projections
            attn = layer.self_attn
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)

            # Reshape to multi-head
            q = q.view(1, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)
            k = k.view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # NO QK norms in Qwen2.5 (unlike Qwen3)

            # Apply standard RoPE
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            # In-place KV cache update
            k_cache[:, :, past_kv_len:end_step, :] = k.half()
            v_cache[:, :, past_kv_len:end_step, :] = v.half()

            k_full = k_cache[:, :, :end_step, :].float()
            v_full = v_cache[:, :, :end_step, :].float()

            # GQA expansion
            k_full = repeat_kv(k_full, self.gqa_repeat)
            v_full = repeat_kv(v_full, self.gqa_repeat)

            # Scaled dot-product attention
            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)

            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(1, q_len, self.num_q_heads * self.head_dim)
            hidden_states = attn.o_proj(attn_output)

            # Residual connection
            hidden_states = residual + hidden_states

            # Post-attention LayerNorm + MLP (SwiGLU)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            mlp = layer.mlp
            gate = mlp.gate_proj(hidden_states)
            up = mlp.up_proj(hidden_states)
            hidden_states = mlp.down_proj(F.silu(gate) * up)

            hidden_states = residual + hidden_states

        # Fused lmHead: slice last position, apply RMSNorm + linear
        last_hidden = hidden_states[:, -1:, :]
        last_hidden = self.final_norm(last_hidden)
        logits = self.lm_head(last_hidden)
        return logits


# ---------------------------------------------------------------------------
# Audio Encoder Wrappers
# ---------------------------------------------------------------------------

class AcousticEncoderWrapper(nn.Module):
    """Wraps the acoustic tokenizer encoder + SpeechConnector for CoreML export.

    The VibeVoice-ASR encoder pipeline (non-streaming, deterministic):
      1. audio [1, num_samples] → unsqueeze(1) → [1, 1, num_samples]
      2. acoustic_tokenizer.encode(audio_4d) → VibeVoiceTokenizerEncoderOutput
      3. .mean → [1, T, acoustic_vae_dim]  (e.g. [1, T, 64])
      4. SpeechConnector (fc1 → RMSNorm → fc2) → [1, T, hidden_size]

    Input:  audio [1, num_samples] — raw 24kHz mono Float32
    Output: acoustic_features [1, T, hidden_size]
    """

    def __init__(self, acoustic_tokenizer, acoustic_connector):
        super().__init__()
        self.tokenizer = acoustic_tokenizer
        self.connector = acoustic_connector

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # acoustic_tokenizer.encode expects [B, 1, num_samples]
        audio_4d = audio.unsqueeze(1)
        encoder_output = self.tokenizer.encode(audio_4d)
        # Use .mean for deterministic output (no VAE sampling)
        features = encoder_output.mean  # [1, T, acoustic_vae_dim]
        return self.connector(features)  # [1, T, hidden_size]


class SemanticEncoderWrapper(nn.Module):
    """Wraps the semantic tokenizer encoder + SpeechConnector for CoreML export.

    Same pipeline as acoustic but with semantic tokenizer (vae_dim=128).

    Input:  audio [1, num_samples] — raw 24kHz mono Float32
    Output: semantic_features [1, T, hidden_size]
    """

    def __init__(self, semantic_tokenizer, semantic_connector):
        super().__init__()
        self.tokenizer = semantic_tokenizer
        self.connector = semantic_connector

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # semantic_tokenizer.encode expects [B, 1, num_samples]
        audio_4d = audio.unsqueeze(1)
        encoder_output = self.tokenizer.encode(audio_4d)
        features = encoder_output.mean  # [1, T, semantic_vae_dim]
        return self.connector(features)  # [1, T, hidden_size]


class FusedAudioEncoderWrapper(nn.Module):
    """Fused encoder: both tokenizers + projector → single output.

    Use this when the audio_projector takes concatenated features.

    Input:  audio [1, num_samples] — raw 24kHz mono audio
    Output: audio_features [1, T, hidden_size] — merged features ready for LLM
    """

    def __init__(self, acoustic_tokenizer, semantic_tokenizer, audio_projector):
        super().__init__()
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        self.audio_projector = audio_projector

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        acoustic_features = self.acoustic_tokenizer.encode(audio)
        semantic_features = self.semantic_tokenizer.encode(audio)

        # The audio_projector merges both feature streams
        audio_features = self.audio_projector(acoustic_features, semantic_features)
        return audio_features


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_full_model(model_id: str):
    """Load the full VibeVoice-ASR model using the vibevoice package.

    Returns a VibeVoiceASRForConditionalGeneration instance loaded in bfloat16
    to fit within 24 GB unified memory (full model ~18 GB in bfloat16).
    """
    # vibevoice package required — install with:
    #   uv pip install --no-deps git+https://github.com/microsoft/VibeVoice.git
    #   uv pip install --no-deps diffusers
    try:
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "The 'vibevoice' package is required to load encoder components.\n"
            "Install it with:\n"
            "  cd Tools && uv pip install --no-deps git+https://github.com/microsoft/VibeVoice.git\n"
            "  uv pip install --no-deps diffusers"
        ) from e

    print(f"Loading model: {model_id} (vibevoice package)")
    t0 = time.time()

    # Load in bfloat16 to stay within ~18 GB on 24 GB unified memory.
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    elapsed = time.time() - t0
    print(f"  Full model loaded in {elapsed:.1f}s. Type: {type(model).__name__}")
    print("\n  model.model children:")
    for name, _ in model.model.named_children():
        print(f"    {name}")
    print()

    # Deep-copy only the small encoder sub-modules (each ~50-200 MB in bfloat16),
    # then immediately free the full 18 GB model before converting to float32.
    # This avoids in-place type conversion that would briefly require both dtypes
    # in memory at the same time for a 9B-parameter model.
    import copy
    print("  Extracting encoder components via deepcopy...")
    t1 = time.time()
    acoustic_tok = copy.deepcopy(model.model.acoustic_tokenizer)
    semantic_tok = copy.deepcopy(model.model.semantic_tokenizer)
    acoustic_conn = copy.deepcopy(model.model.acoustic_connector)
    semantic_conn = copy.deepcopy(model.model.semantic_connector)
    del model
    import gc; gc.collect()
    print(f"  Extracted in {time.time() - t1:.1f}s. Full model freed.")

    # Convert copies to float32 (encoders are small, fast cast)
    acoustic_tok = acoustic_tok.float()
    semantic_tok = semantic_tok.float()
    acoustic_conn = acoustic_conn.float()
    semantic_conn = semantic_conn.float()

    # Return a lightweight namespace with the same .model.* attribute access
    # that convert_audio_encoders() expects.
    from types import SimpleNamespace
    inner = SimpleNamespace(
        acoustic_tokenizer=acoustic_tok,
        semantic_tokenizer=semantic_tok,
        acoustic_connector=acoustic_conn,
        semantic_connector=semantic_conn,
    )
    return SimpleNamespace(model=inner)


def load_decoder_weights(model_id: str, arch: dict):
    """Load text decoder weights into a Qwen2 model from transformers."""
    from transformers import Qwen2Config, Qwen2Model

    print(f"\nLoading decoder weights from: {model_id}")
    t0 = time.time()

    text_config = Qwen2Config(
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["num_layers"],
        num_attention_heads=arch["num_q_heads"],
        num_key_value_heads=arch["num_kv_heads"],
        vocab_size=arch["vocab_size"],
        max_position_embeddings=131072,
        rms_norm_eps=arch["rms_norm_eps"],
        rope_theta=arch["rope_theta"],
        hidden_act="silu",
        attention_bias=True,  # Qwen2.5 uses attention bias
        tie_word_embeddings=False,
    )

    text_model = Qwen2Model(text_config)
    text_model.eval()

    all_weights = load_all_weights(model_id)

    # Map weight keys: VibeVoice uses "model.language_model." prefix for the LLM backbone
    decoder_weights: Dict[str, torch.Tensor] = {}
    for k, v in all_weights.items():
        # Try different prefixes the model might use (VibeVoice first, then legacy)
        for prefix in ["model.language_model.", "llm.model.", "model.llm.model.", "llm."]:
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                # Skip if it's lm_head or embed_tokens (handled separately)
                if not new_key.startswith("lm_head"):
                    decoder_weights[new_key] = v.float()
                break

    missing, unexpected = text_model.load_state_dict(decoder_weights, strict=False)
    print(f"  Weights loaded in {time.time() - t0:.1f}s")
    print(f"  Loaded: {len(decoder_weights)} tensors")
    if missing:
        print(f"  Missing (OK): {len(missing)} — {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} — {unexpected[:5]}...")

    # Extract lm_head weights
    lm_head = nn.Linear(arch["hidden_size"], arch["vocab_size"], bias=False)
    lm_head_candidates = [
        "lm_head.weight",          # VibeVoice: top-level lm_head
        "llm.lm_head.weight",
        "model.lm_head.weight",
    ]
    lm_head_weight = None
    for candidate in lm_head_candidates:
        if candidate in all_weights:
            lm_head_weight = all_weights[candidate].float()
            print(f"  lm_head: loaded from {candidate} ({lm_head_weight.shape})")
            break

    if lm_head_weight is None:
        lm_head_weight = text_model.embed_tokens.weight.data.float()
        print(f"  lm_head: tied to embed_tokens.weight ({lm_head_weight.shape})")
    lm_head.weight = nn.Parameter(lm_head_weight)
    lm_head.eval()

    return text_model, lm_head, all_weights


# ---------------------------------------------------------------------------
# Conversion: Fused Decoder
# ---------------------------------------------------------------------------

def convert_fused_decoder(
    text_model,
    lm_head: nn.Linear,
    arch: dict,
    output_dir: Path,
    max_seq_len: int = 4096,
    skip_validation: bool = False,
) -> Path:
    """Convert the fused Qwen2.5-7B decoder to stateful CoreML."""
    print("\n=== Converting Fused Stateful Decoder (Qwen2.5-7B) ===")
    HIDDEN_SIZE = arch["hidden_size"]
    HEAD_DIM = arch["head_dim"]
    NUM_LAYERS = arch["num_layers"]
    NUM_KV_HEADS = arch["num_kv_heads"]
    VOCAB_SIZE = arch["vocab_size"]

    layers = text_model.layers
    final_norm = text_model.norm
    print(f"  {len(layers)} layers, hidden={HIDDEN_SIZE}, heads={arch['num_q_heads']}/{NUM_KV_HEADS}")
    print(f"  head_dim={HEAD_DIM}, vocab={VOCAB_SIZE}")

    # Verify architecture
    layer0 = layers[0]
    attn0 = layer0.self_attn
    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"
    assert attn0.q_proj.out_features == arch["num_q_heads"] * HEAD_DIM
    assert attn0.k_proj.out_features == NUM_KV_HEADS * HEAD_DIM
    print(f"  q_proj: {attn0.q_proj.in_features} -> {attn0.q_proj.out_features}")
    print(f"  k_proj: {attn0.k_proj.in_features} -> {attn0.k_proj.out_features}")
    print(f"  QK norms: {hasattr(attn0, 'q_norm')} (expected False for Qwen2.5)")
    print(f"  Attention bias: {attn0.q_proj.bias is not None}")

    # Create stateful wrapper
    print(f"  Creating fused stateful decoder (max_seq_len={max_seq_len})...")
    stateful_model = FusedStatefulQwen2Decoder(
        layers, final_norm, lm_head, arch, max_seq_len=max_seq_len
    )
    stateful_model.eval()

    # Trace
    trace_q = 1
    trace_end = 5
    hidden = torch.randn(1, trace_q, HIDDEN_SIZE)
    cos_in = torch.randn(1, trace_q, HEAD_DIM)
    sin_in = torch.randn(1, trace_q, HEAD_DIM)
    mask = torch.zeros(1, 1, trace_q, trace_end)

    print("  Tracing model...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(stateful_model, (hidden, cos_in, sin_in, mask))
    traced.eval()
    print(f"  Trace complete in {time.time() - t0:.1f}s")

    # Validate
    if not skip_validation:
        print("  Validating traced vs eager...")
        stateful_ref = FusedStatefulQwen2Decoder(
            layers, final_norm, lm_head, arch, max_seq_len=max_seq_len
        )
        stateful_ref.eval()
        test_hidden = torch.randn(1, 1, HIDDEN_SIZE)
        test_cos = torch.randn(1, 1, HEAD_DIM)
        test_sin = torch.randn(1, 1, HEAD_DIM)
        test_mask = torch.zeros(1, 1, 1, 3)
        with torch.no_grad():
            ref_out = stateful_ref(test_hidden, test_cos, test_sin, test_mask)
            traced_out = traced(test_hidden, test_cos, test_sin, test_mask)
            diff = (ref_out - traced_out).abs().max().item()
        print(f"  Max diff (traced vs eager): {diff:.6e}")
        if diff > 1e-3:
            print("  WARNING: Large divergence!")
        else:
            print("  OK — traced model matches eager mode")

    # Convert to CoreML
    print("\n  Converting to CoreML...")
    print(f"  coremltools version: {ct.__version__}")

    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_seq_len, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType("logits", dtype=np.float32),
    ]

    states = []
    for i in range(NUM_LAYERS):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM), dtype=np.float16
            ),
            name=f"k_cache_{i}",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM), dtype=np.float16
            ),
            name=f"v_cache_{i}",
        ))

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    print(f"  CoreML conversion complete in {time.time() - t0:.1f}s")

    # Save
    output_path = output_dir / "vibevoice_decoder_stateful.mlpackage"
    mlmodel.save(str(output_path))
    print(f"  Saved: {output_path}")

    # Validate CoreML
    print("  Validating CoreML model...")
    try:
        state = mlmodel.make_state()
        test_input = {
            "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
            "position_cos": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "position_sin": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        output = mlmodel.predict(test_input, state=state)
        out_arr = output["logits"]
        print(f"  Output shape: {out_arr.shape} (expect (1, 1, {VOCAB_SIZE}))")
        print(f"  Output range: [{out_arr.min():.4f}, {out_arr.max():.4f}]")
        print("  CoreML validation passed!")
    except Exception as e:
        print(f"  CoreML validation failed: {e}")
        print("  The model was saved but may need debugging.")

    return output_path


# ---------------------------------------------------------------------------
# Conversion: Audio Encoders
# ---------------------------------------------------------------------------

def convert_audio_encoders(
    model,
    arch: dict,
    output_dir: Path,
    audio_duration_secs: float = 10.0,
) -> dict:
    """Convert acoustic and semantic encoders to CoreML.

    Attempts three strategies:
    1. Fused: single model with both encoders + projector
    2. Separate with projection: each encoder fused with its projector path
    3. Raw: each encoder without projection

    Returns dict of converted model paths.
    """
    print("\n=== Converting Audio Encoders ===")

    # Discover model components
    # VibeVoice-ASR: components live under model.model (VibeVoiceASRModel)
    inner = model.model  # VibeVoiceASRModel
    acoustic_tok = inner.acoustic_tokenizer
    semantic_tok = inner.semantic_tokenizer
    acoustic_conn = inner.acoustic_connector
    semantic_conn = inner.semantic_connector
    print(f"  acoustic_tokenizer: {type(acoustic_tok).__name__}")
    print(f"  semantic_tokenizer: {type(semantic_tok).__name__}")
    print(f"  acoustic_connector: {type(acoustic_conn).__name__}")
    print(f"  semantic_connector: {type(semantic_conn).__name__}")

    num_samples = int(audio_duration_secs * SAMPLE_RATE)
    test_audio = torch.randn(1, num_samples)
    print(f"  Test audio: {audio_duration_secs}s = {num_samples} samples")

    paths = {}

    # Convert acoustic encoder
    print("\n  Converting acoustic encoder...")
    acoustic_tok.eval()
    acoustic_conn.eval()
    try:
        wrapper = AcousticEncoderWrapper(acoustic_tok, acoustic_conn)
        wrapper.eval()

        with torch.no_grad():
            test_out = wrapper(test_audio)
        print(f"  Acoustic output shape: {test_out.shape}")

        path = _convert_single_encoder(
            wrapper, test_audio, "vibevoice_acoustic_encoder",
            "acoustic_features", arch, output_dir
        )
        paths["acoustic_encoder"] = path
    except Exception as e:
        print(f"  Acoustic encoder conversion failed: {e}")
        print("  Attempting raw encoder export...")
        try:
            path = _convert_raw_encoder(
                acoustic_tok, test_audio, "vibevoice_acoustic_encoder",
                "acoustic_features", output_dir
            )
            paths["acoustic_encoder"] = path
        except Exception as e2:
            print(f"  Raw acoustic encoder also failed: {e2}")
            raise

    # Convert semantic encoder
    print("\n  Converting semantic encoder...")
    semantic_tok.eval()
    semantic_conn.eval()
    try:
        wrapper = SemanticEncoderWrapper(semantic_tok, semantic_conn)
        wrapper.eval()

        with torch.no_grad():
            test_out = wrapper(test_audio)
        print(f"  Semantic output shape: {test_out.shape}")

        path = _convert_single_encoder(
            wrapper, test_audio, "vibevoice_semantic_encoder",
            "semantic_features", arch, output_dir
        )
        paths["semantic_encoder"] = path
    except Exception as e:
        print(f"  Semantic encoder conversion failed: {e}")
        try:
            path = _convert_raw_encoder(
                semantic_tok, test_audio, "vibevoice_semantic_encoder",
                "semantic_features", output_dir
            )
            paths["semantic_encoder"] = path
        except Exception as e2:
            print(f"  Raw semantic encoder also failed: {e2}")
            raise

    return paths


def _convert_single_encoder(
    wrapper: nn.Module,
    test_audio: torch.Tensor,
    model_name: str,
    output_name: str,
    arch: dict,
    output_dir: Path,
) -> Path:
    """Trace and convert a single encoder module to CoreML."""
    num_samples = test_audio.shape[-1]

    print(f"  Tracing {model_name}...")
    with torch.no_grad():
        # torch.jit.trace with the fixed input size — the encoder computes
        # `pad_amount = ideal_length - x.shape[-1]` using the input shape.
        # We *freeze* the trace afterwards so that PyTorch's constant-folding
        # pass replaces `aten::size` calls (which coremltools sees as dynamic)
        # with their concrete integer values from the fixed-shape run.
        traced = torch.jit.trace(wrapper, (test_audio,), strict=False)
        traced.eval()
        # freeze() inlines module parameters AND runs constant propagation —
        # this converts `aten::size(input, dim) → prim::Constant[value=N]` for
        # the specific traced shape, removing all "dynamic" shape ops that
        # caused coremltools to complain about dynamic padding.
        try:
            frozen = torch.jit.freeze(traced.eval())
        except Exception as freeze_err:
            print(f"  torch.jit.freeze failed ({freeze_err}), using non-frozen trace...")
            frozen = traced
    model_to_convert = frozen

    # NOTE: The VibeVoice audio tokenizer computes dynamic padding from the input
    # length (ceil(n_frames) arithmetic), so the traced constants are baked for
    # exactly `num_samples`.  CoreML cannot parameterise this with RangeDim —
    # it fails with "Dynamic padding for n-dimensional tensors is not supported".
    # We therefore export a fixed-shape model; callers must chunk / pad to match.
    inputs = [
        ct.TensorType("audio", shape=(1, num_samples), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(output_name, dtype=np.float32),
    ]

    print(f"  Converting {model_name} to CoreML (fixed {num_samples} samples = {num_samples / SAMPLE_RATE:.1f}s)...")
    t0 = time.time()
    coreml_model = ct.convert(
        model_to_convert,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    elapsed = time.time() - t0
    print(f"  CoreML conversion complete in {elapsed:.1f}s")

    path = output_dir / f"{model_name}.mlpackage"
    coreml_model.save(str(path))
    print(f"  Saved: {path}")

    # Validate
    print(f"  Validating {model_name}...")
    try:
        test_np = np.random.randn(1, num_samples).astype(np.float32)
        out = coreml_model.predict({"audio": test_np})
        out_arr = out[output_name]
        print(f"  Output shape: {out_arr.shape}")
        print(f"  Output range: [{out_arr.min():.4f}, {out_arr.max():.4f}]")
        print(f"  {model_name} validation passed!")
    except Exception as e:
        print(f"  {model_name} validation failed: {e}")

    return path


def _convert_raw_encoder(
    encoder: nn.Module,
    test_audio: torch.Tensor,
    model_name: str,
    output_name: str,
    output_dir: Path,
) -> Path:
    """Last-resort: trace the raw encoder module (no wrapper)."""
    num_samples = test_audio.shape[-1]

    # Try to call the encode method directly
    class RawEncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, audio):
            # encode() expects [B, 1, T]; our audio input is [1, T]
            audio_4d = audio.unsqueeze(1)  # [1, T] → [1, 1, T]
            if hasattr(self.enc, "encode"):
                result = self.enc.encode(audio_4d)
                # Returns VibeVoiceTokenizerEncoderOutput — extract .mean [1, T, D]
                return result.mean if hasattr(result, "mean") else result
            return self.enc(audio_4d)

    wrapper = RawEncoderWrapper(encoder)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(test_audio)
    print(f"  Raw encoder output shape: {test_out.shape}")

    traced = torch.jit.trace(wrapper, (test_audio,), strict=False)
    traced.eval()
    try:
        frozen = torch.jit.freeze(traced.eval())
    except Exception:
        frozen = traced

    inputs = [ct.TensorType("audio", shape=(1, num_samples), dtype=np.float32)]
    outputs = [ct.TensorType(output_name, dtype=np.float32)]

    coreml_model = ct.convert(
        frozen, convert_to="mlprogram", inputs=inputs, outputs=outputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    path = output_dir / f"{model_name}.mlpackage"
    coreml_model.save(str(path))
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Extraction: Embeddings
# ---------------------------------------------------------------------------

def extract_embeddings(
    all_weights: dict,
    arch: dict,
    output_dir: Path,
) -> Path:
    """Extract token embeddings as float16 binary."""
    print("\n=== Extracting Embeddings ===")

    # Try various key patterns for embed_tokens
    embed_candidates = [
        "model.language_model.embed_tokens.weight",  # VibeVoice
        "llm.model.embed_tokens.weight",
        "model.llm.model.embed_tokens.weight",
        "llm.embed_tokens.weight",
        "model.embed_tokens.weight",
    ]

    embed_weight = None
    for key in embed_candidates:
        if key in all_weights:
            embed_weight = all_weights[key]
            print(f"  Found embeddings at: {key}")
            break

    if embed_weight is None:
        # Fallback: search for any embed_tokens.weight
        for k, v in all_weights.items():
            if "embed_tokens.weight" in k:
                embed_weight = v
                print(f"  Found embeddings at: {k}")
                break

    if embed_weight is None:
        raise KeyError("Cannot find embed_tokens.weight in model weights")

    vocab_size, hidden_size = embed_weight.shape
    print(f"  Embedding shape: [{vocab_size}, {hidden_size}]")

    assert vocab_size == arch["vocab_size"], f"Vocab mismatch: {vocab_size} != {arch['vocab_size']}"
    assert hidden_size == arch["hidden_size"], f"Hidden mismatch: {hidden_size} != {arch['hidden_size']}"

    # Convert to float16
    embed_f16 = embed_weight.half().numpy()

    # Write binary: uint32 vocab_size, uint32 hidden_size, then float16[vocab_size * hidden_size]
    path = output_dir / "vibevoice_embeddings.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_size))
        f.write(embed_f16.tobytes())

    file_size = path.stat().st_size
    expected_size = 8 + vocab_size * hidden_size * 2
    print(f"  File size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")
    print(f"  Expected:  {expected_size:,} bytes")
    assert file_size == expected_size, f"File size mismatch!"
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Copy: Vocabulary
# ---------------------------------------------------------------------------

def copy_vocabulary(model_id: str, output_dir: Path) -> Path:
    """Copy vocab.json from the tokenizer.

    VibeVoice-ASR does not ship its own tokenizer files — it uses the
    Qwen2.5-7B tokenizer. We try the model repo first, then fall back to
    Qwen/Qwen2.5-7B.
    """
    from huggingface_hub import hf_hub_download

    print("\n=== Copying Vocabulary ===")

    # Try the model repo first (some fine-tunes ship their own vocab)
    for source_id in [model_id, "Qwen/Qwen2.5-7B"]:
        for filename in ["vocab.json", "tokenizer.json"]:
            try:
                vocab_path = hf_hub_download(source_id, filename)
                if filename == "vocab.json":
                    dest = output_dir / "vocab.json"
                    shutil.copy2(vocab_path, dest)
                    with open(dest) as f:
                        vocab = json.load(f)
                    print(f"  Vocabulary: {len(vocab)} tokens (from {source_id}/{filename})")
                    print(f"  Saved: {dest}")
                    return dest
                else:  # tokenizer.json — extract vocab
                    with open(vocab_path) as f:
                        tokenizer_data = json.load(f)
                    vocab: dict = {}
                    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                        raw_vocab = tokenizer_data["model"]["vocab"]
                        if isinstance(raw_vocab, dict):
                            vocab = raw_vocab
                        elif isinstance(raw_vocab, list):
                            vocab = {token: idx for idx, token in enumerate(raw_vocab)}
                    if "added_tokens" in tokenizer_data:
                        for token_info in tokenizer_data["added_tokens"]:
                            if isinstance(token_info, dict):
                                vocab[token_info["content"]] = token_info["id"]
                    dest = output_dir / "vocab.json"
                    with open(dest, "w") as f:
                        json.dump(vocab, f, ensure_ascii=False)
                    print(f"  Vocabulary: {len(vocab)} tokens (from {source_id}/{filename})")
                    print(f"  Saved: {dest}")
                    return dest
            except Exception:
                continue

    raise FileNotFoundError(
        f"Could not find tokenizer files in {model_id} or Qwen/Qwen2.5-7B"
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata(
    model_id: str,
    arch: dict,
    output_dir: Path,
    component_paths: dict,
    max_seq_len: int,
) -> Path:
    """Write metadata.json documenting all exported components."""
    metadata = {
        "model_id": model_id,
        "architecture": "VibeVoiceASRForConditionalGeneration",
        "sample_rate": SAMPLE_RATE,
        "compression_ratio": COMPRESSION_RATIO,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "max_seq_length": max_seq_len,
        "acoustic_tokenizer": {
            "latent_dim": arch["acoustic_latent_dim"],
            "d_model": arch["acoustic_d_model"],
            "encoder_strides": arch["acoustic_strides"],
        },
        "semantic_tokenizer": {
            "latent_dim": arch["semantic_latent_dim"],
            "d_model": arch["semantic_d_model"],
        },
        "audio_projector": {
            "hidden_size": arch["audio_projector_hidden_size"],
        },
        "text_decoder": {
            "backbone": "Qwen2.5-7B",
            "hidden_size": arch["hidden_size"],
            "intermediate_size": arch["intermediate_size"],
            "num_layers": arch["num_layers"],
            "num_attention_heads": arch["num_q_heads"],
            "num_kv_heads": arch["num_kv_heads"],
            "head_dim": arch["head_dim"],
            "vocab_size": arch["vocab_size"],
            "rope_theta": arch["rope_theta"],
            "qk_norms": False,  # Qwen2.5, not Qwen3
        },
        "special_tokens": {
            "speech_start_token_id": 151852,
            "speech_end_token_id": 151853,
            "speech_pad_token_id": 151854,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "eos_token_ids": [151643, 151645],
        },
        "components": {k: {"path": str(v.name) if hasattr(v, 'name') else str(v)} for k, v in component_paths.items()},
        "export_settings": {
            "decoder_compute_units": "CPU_AND_GPU",
            "decoder_compute_precision": "FLOAT16",
            "encoder_compute_units": "CPU_ONLY",
            "deployment_target": "macOS15/iOS18",
        },
    }

    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nMetadata written to {path}")
    return path


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert VibeVoice-ASR to CoreML for on-device Apple inference"
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: auto-detected from model ID)"
    )
    parser.add_argument(
        "--components", default=None,
        help="Comma-separated: encoders,decoder,embeddings,vocab (default: all)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=4096,
        help="Max sequence length for KV cache (default: 4096)"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip PyTorch validation steps"
    )
    args = parser.parse_args()

    model_id = args.model_id
    max_seq_len = args.max_seq_len

    # Auto-detect output directory
    if args.output_dir is None:
        model_name = model_id.split("/")[-1].lower().replace(" ", "-")
        output_dir = Path(f"build/{model_name}-coreml")
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse components
    if args.components:
        convert_list = [c.strip() for c in args.components.split(",")]
    else:
        convert_list = ["encoders", "decoder", "embeddings", "vocab"]

    print(f"Model:      {model_id}")
    print(f"Output:     {output_dir}")
    print(f"Components: {convert_list}")
    print(f"Max seq:    {max_seq_len}")

    # Load config
    print("\nLoading model config...")
    config = load_model_config(model_id)
    arch = get_arch_constants(config)
    print(f"  Acoustic: latent_dim={arch['acoustic_latent_dim']}, strides={arch['acoustic_strides']}")
    print(f"  Semantic: latent_dim={arch['semantic_latent_dim']}")
    print(f"  Decoder:  hidden={arch['hidden_size']}, layers={arch['num_layers']}, "
          f"heads={arch['num_q_heads']}/{arch['num_kv_heads']}, vocab={arch['vocab_size']}")

    component_paths = {}

    # Audio encoders need the full model loaded via trust_remote_code
    if "encoders" in convert_list:
        full_model = load_full_model(model_id)
        paths = convert_audio_encoders(full_model, arch, output_dir)
        component_paths.update(paths)
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    # Decoder uses transformers Qwen2Model directly (no custom code needed)
    if "decoder" in convert_list:
        text_model, lm_head, all_weights = load_decoder_weights(model_id, arch)
        path = convert_fused_decoder(
            text_model, lm_head, arch, output_dir,
            max_seq_len=max_seq_len,
            skip_validation=args.skip_validation,
        )
        component_paths["decoder_stateful"] = path

        # Extract embeddings from the already-loaded weights
        if "embeddings" in convert_list:
            path = extract_embeddings(all_weights, arch, output_dir)
            component_paths["embeddings"] = path
            convert_list = [c for c in convert_list if c != "embeddings"]

        del text_model, lm_head, all_weights
        import gc; gc.collect()
    elif "embeddings" in convert_list:
        all_weights = load_all_weights(model_id)
        path = extract_embeddings(all_weights, arch, output_dir)
        component_paths["embeddings"] = path
        del all_weights
        import gc; gc.collect()

    if "vocab" in convert_list:
        path = copy_vocabulary(model_id, output_dir)
        component_paths["vocab"] = path

    # Write metadata
    write_metadata(model_id, arch, output_dir, component_paths, max_seq_len)

    print("\n=== Conversion complete ===")
    print(f"Output directory: {output_dir}")
    for name, path in component_paths.items():
        print(f"  {name}: {path}")
    print(f"\nTotal disk size: {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert Qwen3-ASR (0.6B or 1.7B) to CoreML for on-device Apple inference.

Produces the 2-model pipeline used by FluidAudio's Qwen3AsrManager:
  1. qwen3_asr_audio_encoder.mlpackage   — mel spectrogram → audio features
  2. qwen3_asr_decoder_stateful.mlpackage — stateful decoder with fused lmHead
  3. qwen3_asr_embeddings.bin             — float16 token embedding matrix
  4. vocab.json                           — tokenizer vocabulary
  5. metadata.json                        — model configuration

Usage:
    # Convert 1.7B model (default)
    uv run convert_qwen3_asr_to_coreml.py

    # Convert 0.6B model
    uv run convert_qwen3_asr_to_coreml.py --model-id Qwen/Qwen3-ASR-0.6B

    # Convert only specific components
    uv run convert_qwen3_asr_to_coreml.py --components decoder
    uv run convert_qwen3_asr_to_coreml.py --components encoder
    uv run convert_qwen3_asr_to_coreml.py --components embeddings,vocab

    # Custom output directory
    uv run convert_qwen3_asr_to_coreml.py --output-dir ./build/qwen3-asr-1.7b

Architecture:
  Qwen3ASRForConditionalGeneration
    └── thinker
          ├── audio_tower  → qwen3_asr_audio_encoder.mlpackage
          ├── model         → (fused into decoder_stateful)
          │   ├── embed_tokens → qwen3_asr_embeddings.bin (float16)
          │   ├── layers[0..N] → decoder layers with KV cache states
          │   └── norm         → fused into decoder_stateful
          └── lm_head       → fused into decoder_stateful

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

# Defaults for 1.7B (overridden by --model-id)
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# Shared constants
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
MEL_WINDOW_SIZE = 100  # n_window * 2
CONV_DOWNSAMPLE_FACTOR = 8
MAX_AUDIO_SECONDS = 30.0


def load_model_config(model_id: str) -> dict:
    """Load and parse architecture config from HuggingFace."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return config


def get_arch_constants(config: dict) -> dict:
    """Extract architecture constants from config.json."""
    # Audio encoder config
    audio_cfg = config.get("audio_config", config.get("audio_encoder", {}))
    # Text decoder config
    text_cfg = config.get("text_config", config.get("thinker_config", {}))

    return {
        # Audio encoder
        "encoder_d_model": audio_cfg.get("d_model", 1024),
        "encoder_num_layers": audio_cfg.get("encoder_layers", 24),
        "encoder_num_heads": audio_cfg.get("encoder_attention_heads", 16),
        "encoder_ffn_dim": audio_cfg.get("encoder_ffn_dim", 4096),
        "encoder_output_dim": audio_cfg.get("output_dim", 2048),
        "downsample_hidden_size": audio_cfg.get("downsample_hidden_size", 480),
        # Text decoder
        "hidden_size": text_cfg.get("hidden_size", 2048),
        "intermediate_size": text_cfg.get("intermediate_size", 6144),
        "num_layers": text_cfg.get("num_hidden_layers", 28),
        "num_q_heads": text_cfg.get("num_attention_heads", 16),
        "num_kv_heads": text_cfg.get("num_key_value_heads", 8),
        "head_dim": text_cfg.get("head_dim", 128),
        "vocab_size": text_cfg.get("vocab_size", 151_936),
        "rope_theta": text_cfg.get("rope_theta", 1_000_000),
        "rms_norm_eps": text_cfg.get("rms_norm_eps", 1e-6),
        "mrope_section": text_cfg.get("mrope_section", [24, 20, 20]),
    }


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_all_weights(model_id: str) -> dict:
    """Download and load all safetensors weights, handling multi-shard models."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Check if model has a weight index (multi-shard)
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
        # Single shard
        st_path = hf_hub_download(model_id, "model.safetensors")
        return load_file(st_path)


# ---------------------------------------------------------------------------
# Fused Stateful Decoder (decoder + norm + lm_head)
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


class FusedStatefulQwen3Decoder(nn.Module):
    """Qwen3-ASR decoder with fused lmHead and stateful KV cache for CoreML export.

    Wraps the transformer decoder layers, final RMSNorm, and lm_head.
    Outputs logits for the last query position only.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Linear,
        arch: dict,
        max_seq_len: int = 512,
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

        cos = position_cos.unsqueeze(1)
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

            # QK norms (Qwen3 feature)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            # Apply RoPE
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
# Audio Encoder Wrapper
# ---------------------------------------------------------------------------

class AudioEncoderWrapper(nn.Module):
    """Full audio encoder forward pass for CoreML with fixed-size input.

    Processes a single mel spectrogram window through:
    1. Conv2D downsampling (3 layers, stride 2 each → 8x time reduction)
    2. Linear projection (c*f → d_model)
    3. Sinusoidal positional embedding
    4. N transformer encoder layers
    5. LayerNorm → proj1 → GELU → proj2

    Input:  mel_input [1, 128, T]  mel spectrogram
    Output: features  [1, T', output_dim]
    """

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.conv2d1 = audio_encoder.conv2d1
        self.conv2d2 = audio_encoder.conv2d2
        self.conv2d3 = audio_encoder.conv2d3
        self.conv_out = audio_encoder.conv_out
        self.positional_embedding = audio_encoder.positional_embedding
        self.layers = audio_encoder.layers
        self.ln_post = audio_encoder.ln_post
        self.proj1 = audio_encoder.proj1
        self.proj2 = audio_encoder.proj2

    def forward(self, mel_input: torch.Tensor) -> torch.Tensor:
        # mel_input: [1, 128, T]
        x = mel_input.unsqueeze(1)  # [1, 1, 128, T]

        # Conv downsampling
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x: [B, C, F', T']
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.conv_out(x)  # [B, T', d_model]

        # Positional embedding
        pos_emb = self.positional_embedding(t)  # [T', d_model]
        x = x + pos_emb.unsqueeze(0).to(x.dtype)

        # Transformer layers
        hs = x.squeeze(0)  # [T', d_model]
        seq_len = hs.shape[0]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=hs.device)

        for layer in self.layers:
            layer_outputs = layer(hs, cu_seqlens=cu_seqlens)
            hs = layer_outputs[0]

        hs = self.ln_post(hs)
        hs = self.proj1(hs)
        hs = F.gelu(hs)
        hs = self.proj2(hs)  # [T', output_dim]
        return hs.unsqueeze(0)  # [1, T', output_dim]


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_full_model(model_id: str):
    """Load the full Qwen3-ASR model via trust_remote_code=True."""
    from transformers import AutoModel, AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # Patch ROPE_INIT_FUNCTIONS for transformers 5.x compatibility
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None, **kwargs):
            base = config.rope_theta
            dim = config.head_dim
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
        print("  Patched ROPE_INIT_FUNCTIONS: added 'default' rope type")

    print(f"Loading model: {model_id} (trust_remote_code=True)")
    t0 = time.time()

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Patch missing attributes
    def _ensure_attr(cfg, attr, default):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    _ensure_attr(config, "pad_token_id", None)
    if hasattr(config, "thinker_config"):
        _ensure_attr(config.thinker_config, "pad_token_id", None)

    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s. Type: {type(model).__name__}")
    return model


def load_decoder_weights(model_id: str, arch: dict):
    """Load text decoder weights into a Qwen3Model from transformers."""
    from transformers import Qwen3Config, Qwen3Model

    print(f"\nLoading decoder weights from: {model_id}")
    t0 = time.time()

    text_config = Qwen3Config(
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["num_layers"],
        num_attention_heads=arch["num_q_heads"],
        num_key_value_heads=arch["num_kv_heads"],
        head_dim=arch["head_dim"],
        vocab_size=arch["vocab_size"],
        max_position_embeddings=65_536,
        rms_norm_eps=arch["rms_norm_eps"],
        rope_theta=arch["rope_theta"],
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=True,
    )

    text_model = Qwen3Model(text_config)
    text_model.eval()

    all_weights = load_all_weights(model_id)

    decoder_weights = {}
    for k, v in all_weights.items():
        if k.startswith("thinker.model."):
            new_key = k[len("thinker.model."):]
            decoder_weights[new_key] = v.float()

    missing, unexpected = text_model.load_state_dict(decoder_weights, strict=False)
    print(f"  Weights loaded in {time.time() - t0:.1f}s")
    print(f"  Loaded: {len(decoder_weights)} tensors")
    if missing:
        print(f"  Missing (OK): {len(missing)} — {missing[:3]}...")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} — {unexpected[:3]}...")

    # Extract lm_head weights
    lm_head = nn.Linear(arch["hidden_size"], arch["vocab_size"], bias=False)
    if "thinker.lm_head.weight" in all_weights:
        lm_head_weight = all_weights["thinker.lm_head.weight"].float()
        print(f"  lm_head: loaded from thinker.lm_head.weight ({lm_head_weight.shape})")
    else:
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
    max_seq_len: int = 512,
    skip_validation: bool = False,
) -> Path:
    """Convert the fused decoder (layers + norm + lm_head) to stateful CoreML."""
    print("\n=== Converting Fused Stateful Decoder ===")
    HIDDEN_SIZE = arch["hidden_size"]
    HEAD_DIM = arch["head_dim"]
    NUM_LAYERS = arch["num_layers"]
    NUM_KV_HEADS = arch["num_kv_heads"]
    VOCAB_SIZE = arch["vocab_size"]

    layers = text_model.layers
    final_norm = text_model.norm
    print(f"  {len(layers)} layers, hidden={HIDDEN_SIZE}, heads={arch['num_q_heads']}/{NUM_KV_HEADS}")

    # Verify architecture
    layer0 = layers[0]
    attn0 = layer0.self_attn
    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"
    assert attn0.q_proj.out_features == arch["num_q_heads"] * HEAD_DIM
    assert attn0.k_proj.out_features == NUM_KV_HEADS * HEAD_DIM
    print(f"  q_proj: {attn0.q_proj.in_features} -> {attn0.q_proj.out_features}")
    print(f"  k_proj: {attn0.k_proj.in_features} -> {attn0.k_proj.out_features}")
    print(f"  QK norms: {hasattr(attn0, 'q_norm')}")

    # Create stateful wrapper
    print(f"  Creating fused stateful decoder (max_seq_len={max_seq_len})...")
    stateful_model = FusedStatefulQwen3Decoder(
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
        stateful_ref = FusedStatefulQwen3Decoder(
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
    output_path = output_dir / "qwen3_asr_decoder_stateful.mlpackage"
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
# Conversion: Audio Encoder
# ---------------------------------------------------------------------------

def convert_audio_encoder(
    model,
    arch: dict,
    output_dir: Path,
) -> Path:
    """Convert the audio encoder to CoreML."""
    print("\n=== Converting Audio Encoder ===")

    # Extract audio_tower from model hierarchy
    if hasattr(model, "thinker"):
        audio_encoder = model.thinker.audio_tower
    elif hasattr(model, "audio_tower"):
        audio_encoder = model.audio_tower
    else:
        raise AttributeError("Cannot find audio_tower in model")

    audio_encoder.eval()
    wrapper = AudioEncoderWrapper(audio_encoder)
    wrapper.eval()

    # Trace with single window: 100 mel frames
    mel_input = torch.randn(1, NUM_MEL_BINS, MEL_WINDOW_SIZE, dtype=torch.float32)
    print(f"  Trace input shape: {mel_input.shape}")

    with torch.inference_mode():
        ref_output = wrapper(mel_input)
        print(f"  Reference output shape: {ref_output.shape}")

    mel_input = mel_input.clone()
    print("  Tracing audio encoder...")
    traced = torch.jit.trace(wrapper, (mel_input,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="mel_input",
            shape=(1, NUM_MEL_BINS, MEL_WINDOW_SIZE),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="audio_features", dtype=np.float32),
    ]

    print("  Converting to CoreML...")
    t0 = time.time()
    coreml_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    print(f"  CoreML conversion complete in {time.time() - t0:.1f}s")

    path = output_dir / "qwen3_asr_audio_encoder.mlpackage"
    coreml_model.save(str(path))
    print(f"  Saved: {path}")

    # Validate
    print("  Validating CoreML model...")
    try:
        test_mel = np.random.randn(1, NUM_MEL_BINS, MEL_WINDOW_SIZE).astype(np.float32)
        out = coreml_model.predict({"mel_input": test_mel})
        out_arr = out["audio_features"]
        print(f"  Output shape: {out_arr.shape}")
        print(f"  Output range: [{out_arr.min():.4f}, {out_arr.max():.4f}]")
        print("  CoreML validation passed!")
    except Exception as e:
        print(f"  CoreML validation failed: {e}")

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

    # Get embed_tokens weight
    embed_key = "thinker.model.embed_tokens.weight"
    if embed_key not in all_weights:
        raise KeyError(f"Cannot find {embed_key} in weights")

    embed_weight = all_weights[embed_key]
    vocab_size, hidden_size = embed_weight.shape
    print(f"  Embedding shape: [{vocab_size}, {hidden_size}]")

    assert vocab_size == arch["vocab_size"], f"Vocab mismatch: {vocab_size} != {arch['vocab_size']}"
    assert hidden_size == arch["hidden_size"], f"Hidden mismatch: {hidden_size} != {arch['hidden_size']}"

    # Convert to float16
    embed_f16 = embed_weight.half().numpy()

    # Write binary: uint32 vocab_size, uint32 hidden_size, then float16[vocab_size * hidden_size]
    path = output_dir / "qwen3_asr_embeddings.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", hidden_size))
        f.write(embed_f16.tobytes())

    file_size = path.stat().st_size
    expected_size = 8 + vocab_size * hidden_size * 2
    print(f"  File size: {file_size:,} bytes (expected {expected_size:,})")
    assert file_size == expected_size, f"File size mismatch!"
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Copy: Vocabulary
# ---------------------------------------------------------------------------

def copy_vocabulary(model_id: str, output_dir: Path) -> Path:
    """Copy vocab.json from the tokenizer."""
    from huggingface_hub import hf_hub_download

    print("\n=== Copying Vocabulary ===")
    vocab_path = hf_hub_download(model_id, "vocab.json")
    dest = output_dir / "vocab.json"
    shutil.copy2(vocab_path, dest)

    with open(dest) as f:
        vocab = json.load(f)
    print(f"  Vocabulary: {len(vocab)} tokens")
    print(f"  Saved: {dest}")
    return dest


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
        "architecture": "Qwen3ASRForConditionalGeneration",
        "sample_rate": SAMPLE_RATE,
        "num_mel_bins": NUM_MEL_BINS,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "max_seq_length": max_seq_len,
        "audio_encoder": {
            "n_window": 50,
            "n_window_infer": 800,
            "mel_window_size": MEL_WINDOW_SIZE,
            "conv_downsample_factor": CONV_DOWNSAMPLE_FACTOR,
            "d_model": arch["encoder_d_model"],
            "output_dim": arch["encoder_output_dim"],
            "num_layers": arch["encoder_num_layers"],
            "num_heads": arch["encoder_num_heads"],
        },
        "text_decoder": {
            "hidden_size": arch["hidden_size"],
            "intermediate_size": arch["intermediate_size"],
            "num_layers": arch["num_layers"],
            "num_attention_heads": arch["num_q_heads"],
            "num_kv_heads": arch["num_kv_heads"],
            "head_dim": arch["head_dim"],
            "vocab_size": arch["vocab_size"],
            "rope_theta": arch["rope_theta"],
            "mrope_section": arch["mrope_section"],
        },
        "special_tokens": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "system_token_id": 8948,
            "user_token_id": 872,
            "assistant_token_id": 77091,
            "newline_token_id": 198,
            "eos_token_ids": [151645, 151643],
        },
        "components": component_paths,
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
        description="Convert Qwen3-ASR to CoreML for on-device Apple inference"
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID (default: Qwen/Qwen3-ASR-1.7B)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: auto-detected from model ID)"
    )
    parser.add_argument(
        "--components", default=None,
        help="Comma-separated: encoder,decoder,embeddings,vocab (default: all)"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Max sequence length for KV cache (default: 512)"
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
        convert_list = ["encoder", "decoder", "embeddings", "vocab"]

    print(f"Model:      {model_id}")
    print(f"Output:     {output_dir}")
    print(f"Components: {convert_list}")
    print(f"Max seq:    {max_seq_len}")

    # Load config
    print("\nLoading model config...")
    config = load_model_config(model_id)
    arch = get_arch_constants(config)
    print(f"  Encoder: d_model={arch['encoder_d_model']}, layers={arch['encoder_num_layers']}, "
          f"output_dim={arch['encoder_output_dim']}")
    print(f"  Decoder: hidden={arch['hidden_size']}, intermediate={arch['intermediate_size']}, "
          f"layers={arch['num_layers']}, heads={arch['num_q_heads']}/{arch['num_kv_heads']}")

    component_paths = {}

    # Audio encoder needs the full model loaded via trust_remote_code
    if "encoder" in convert_list:
        full_model = load_full_model(model_id)
        path = convert_audio_encoder(full_model, arch, output_dir)
        component_paths["audio_encoder"] = {"path": path.name}
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    # Decoder uses transformers Qwen3Model directly (no custom code needed)
    if "decoder" in convert_list:
        text_model, lm_head, all_weights = load_decoder_weights(model_id, arch)
        path = convert_fused_decoder(
            text_model, lm_head, arch, output_dir,
            max_seq_len=max_seq_len,
            skip_validation=args.skip_validation,
        )
        component_paths["decoder_stateful"] = {
            "path": path.name,
            "num_layers": arch["num_layers"],
            "fused_lm_head": True,
        }

        # Extract embeddings from the already-loaded weights
        if "embeddings" in convert_list:
            path = extract_embeddings(all_weights, arch, output_dir)
            component_paths["embeddings"] = {"path": path.name}
            convert_list = [c for c in convert_list if c != "embeddings"]  # Mark done

        del text_model, lm_head, all_weights
        import gc; gc.collect()
    elif "embeddings" in convert_list:
        # Load weights just for embeddings
        all_weights = load_all_weights(model_id)
        path = extract_embeddings(all_weights, arch, output_dir)
        component_paths["embeddings"] = {"path": path.name}
        del all_weights
        import gc; gc.collect()

    if "vocab" in convert_list:
        path = copy_vocabulary(model_id, output_dir)
        component_paths["vocab"] = {"path": path.name}

    # Write metadata
    write_metadata(model_id, arch, output_dir, component_paths, max_seq_len)

    print("\n=== Conversion complete ===")
    print(f"Output directory: {output_dir}")
    for name, info in component_paths.items():
        print(f"  {name}: {info['path']}")


if __name__ == "__main__":
    main()

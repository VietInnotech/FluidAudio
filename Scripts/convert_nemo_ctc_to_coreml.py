#!/usr/bin/env python3
"""
Convert nvidia/parakeet-ctc-0.6b-Vietnamese NeMo model to CoreML.

Exports two staged CoreML models + vocab.json matching FluidAudio's CTC model format:
  - MelSpectrogram.mlpackage  → MelSpectrogram.mlmodelc (after xcrun compile)
  - AudioEncoder.mlpackage    → AudioEncoder.mlmodelc   (after xcrun compile)
  - vocab.json

Architecture:
  MelSpectrogram: audio [1, 240000] + audio_length [1] → melspectrogram_features + mel_length
  AudioEncoder:   melspectrogram_features + mel_length [1] → ctc_head_raw_output [1, T, V]

Requirements (use the UV venv in Tools/.venv):
    source Tools/.venv/bin/activate
    # already installed via: uv sync --no-install-project

Usage:
    python convert_nemo_ctc_to_coreml.py
    python convert_nemo_ctc_to_coreml.py --model nvidia/parakeet-ctc-0.6b-Vietnamese --output ./output
    python convert_nemo_ctc_to_coreml.py --model /path/to/model.nemo --output ./output
    python convert_nemo_ctc_to_coreml.py --skip-validation   # skip PyTorch vs CoreML diff check

After conversion, compile on macOS with:
    xcrun coremlcompiler compile output/MelSpectrogram.mlpackage output/
    xcrun coremlcompiler compile output/AudioEncoder.mlpackage output/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

# Disable NeMo's @typecheck decorator that requires kwargs-only calls,
# which is incompatible with torch.jit.trace (passes positional args).
try:
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(enabled=False)
    print("NeMo typecheck disabled for tracing compatibility")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants matching FluidAudio's ASRConstants
# ---------------------------------------------------------------------------
MAX_AUDIO_SAMPLES = 240_000   # 15s at 16kHz
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Wrapper modules for tracing
# ---------------------------------------------------------------------------

class MelSpectrogramWrapper(nn.Module):
    """Wraps NeMo's preprocessor for fixed-length CoreML export.

    Input:  audio       [1, MAX_AUDIO_SAMPLES]  float32
            audio_length [1]                    int32
    Output: melspectrogram_features  [1, n_mels, T_mel]  float32
            mel_length               [1]                  int32
    """
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio: torch.Tensor, audio_length: torch.Tensor) -> tuple:
        mel, mel_len = self.preprocessor(input_signal=audio, length=audio_length)
        return mel, mel_len


class AudioEncoderWrapper(nn.Module):
    """Wraps NeMo's encoder + CTC decoder for CoreML export.

    Input:  melspectrogram_features  [1, n_mels, T_mel]  float32
            mel_length               [1]                  int32
    Output: ctc_head_raw_output      [1, T_enc, vocab_size]  float32 (raw logits, no softmax)
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Access the raw linear projection layer directly to skip log_softmax.
        # ConvASRDecoder.forward() always applies log_softmax which degrades fp16 precision.
        # We output raw logits — argmax is invariant to monotonic transforms.
        self.decoder_layers = decoder.decoder_layers

    def forward(self, melspectrogram_features: torch.Tensor, mel_length: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(audio_signal=melspectrogram_features, length=mel_length)
        # Raw logits without log_softmax: decoder_layers(encoded) → [B, V, T] → transpose to [B, T, V]
        raw_logits = self.decoder_layers(encoded)  # [B, V, T]
        logits = raw_logits.transpose(1, 2)  # [B, T, V]
        return logits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_nemo_model(model_path: str):
    """Load NeMo ASR CTC BPE model from HuggingFace hub or local .nemo file."""
    import nemo.collections.asr as nemo_asr

    print(f"Loading model: {model_path}")

    if model_path.endswith(".nemo") and os.path.isfile(model_path):
        # Direct .nemo file path
        nemo_file = model_path
    else:
        # HuggingFace model name: download the .nemo file directly via hf_hub_download
        # NeMo's from_pretrained has a bug where it passes the download directory
        # to restore_from instead of the .nemo file itself.
        from huggingface_hub import hf_hub_download, list_repo_files
        print(f"  Searching for .nemo file in repo '{model_path}'...")

        # Find the .nemo file in the repo
        nemo_filename = None
        for fname in list_repo_files(model_path):
            if fname.endswith(".nemo"):
                nemo_filename = fname
                print(f"  Found: {nemo_filename}")
                break

        if nemo_filename is None:
            raise RuntimeError(f"No .nemo file found in HuggingFace repo: {model_path}")

        nemo_file = hf_hub_download(repo_id=model_path, filename=nemo_filename)
        print(f"  Downloaded to: {nemo_file}")

    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(nemo_file)
    model.eval()
    model.freeze()

    vocab_size = model.decoder.num_classes_with_blank
    print(f"  Vocab size (with blank): {vocab_size}")
    print(f"  Sample rate:             {model.preprocessor._sample_rate}")
    return model


# ---------------------------------------------------------------------------
# CoreML conversion helpers
# ---------------------------------------------------------------------------

def _trace_module(wrapper: nn.Module, example_inputs: tuple, model_name: str):
    """Trace a wrapped nn.Module to TorchScript."""
    print(f"  Tracing {model_name}...")
    # Disable NeMo type-checking decorators during tracing
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    print(f"  Traced {model_name} successfully")
    return traced


def convert_mel_spectrogram(preprocessor, output_path: str) -> ct.models.MLModel:
    """Convert NeMo preprocessor to CoreML MelSpectrogram model."""
    print("\n[1/4] Converting MelSpectrogram...")

    # Fixed-length dummy input matching FluidAudio's padding approach
    dummy_audio = torch.zeros(1, MAX_AUDIO_SAMPLES)
    dummy_length = torch.tensor([MAX_AUDIO_SAMPLES], dtype=torch.int32)

    # Run once to determine output shape
    with torch.no_grad():
        mel_out, mel_len = preprocessor(dummy_audio, dummy_length)
    mel_shape = mel_out.shape
    print(f"  Mel output shape: {mel_shape}  (mel_length: {mel_len.item()})")

    wrapper = MelSpectrogramWrapper(preprocessor)
    traced = _trace_module(wrapper, (dummy_audio, dummy_length), "MelSpectrogram")

    # CoreML input types
    inputs = [
        ct.TensorType(name="audio", shape=ct.Shape(shape=(1, MAX_AUDIO_SAMPLES)), dtype=np.float32),
        ct.TensorType(name="audio_length", shape=ct.Shape(shape=(1,)), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="melspectrogram_features", dtype=np.float32),
        ct.TensorType(name="mel_length", dtype=np.int32),
    ]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.save(output_path)
    print(f"  Saved: {output_path}")
    return mlmodel


def convert_audio_encoder(encoder, decoder, preprocessor, output_path: str) -> ct.models.MLModel:
    """Convert NeMo encoder + CTC decoder to CoreML AudioEncoder model."""
    print("\n[2/4] Converting AudioEncoder (encoder + CTC head)...")

    # Derive mel shape from preprocessor
    dummy_audio = torch.zeros(1, MAX_AUDIO_SAMPLES)
    dummy_length = torch.tensor([MAX_AUDIO_SAMPLES], dtype=torch.int32)
    with torch.no_grad():
        mel_features, mel_len_tensor = preprocessor(dummy_audio, dummy_length)

    mel_shape = list(mel_features.shape)  # e.g. [1, 80, 1499]
    n_mels = mel_shape[1]
    t_mel = mel_shape[2]
    print(f"  Encoder input shape: {mel_shape}")

    wrapper = AudioEncoderWrapper(encoder, decoder)

    dummy_mel = mel_features
    dummy_mel_len = mel_len_tensor.to(torch.int32)
    traced = _trace_module(wrapper, (dummy_mel, dummy_mel_len), "AudioEncoder")

    # Verify output is raw logits (not log-softmax)
    with torch.no_grad():
        test_out = traced(dummy_mel, dummy_mel_len)
    print(f"  Encoder output shape: {test_out.shape}  (should be [1, T, vocab_size])")

    out_min, out_max = test_out.min().item(), test_out.max().item()
    print(f"  Output value range: [{out_min:.3f}, {out_max:.3f}]  (raw logits, log_softmax disabled)")
    if out_max < 0:
        print("  WARNING: All values negative — log_softmax may still be active")

    inputs = [
        ct.TensorType(name="melspectrogram_features", shape=ct.Shape(shape=mel_shape), dtype=np.float32),
        ct.TensorType(name="mel_length", shape=ct.Shape(shape=(1,)), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="ctc_head_raw_output", dtype=np.float32),
    ]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.save(output_path)
    print(f"  Saved: {output_path}")
    return mlmodel


# ---------------------------------------------------------------------------
# Vocabulary export
# ---------------------------------------------------------------------------

def export_vocabulary(model, output_dir: Path) -> dict:
    """Export tokenizer vocabulary as vocab.json (token_id: token_string)."""
    print("\n[3/4] Exporting vocabulary...")

    vocab_path = output_dir / "vocab.json"
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size

    vocab = {}
    for i in range(vocab_size):
        tokens = tokenizer.ids_to_tokens([i])
        if tokens:
            token = tokens[0] if isinstance(tokens, list) else str(tokens)
        else:
            token = f"<unk_{i}>"
        vocab[str(i)] = token

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {vocab_path}  ({len(vocab)} tokens)")
    print(f"  Sample tokens: {dict(list(vocab.items())[:8])}")
    return vocab


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_conversion(
    model,
    mel_mlpackage: str,
    encoder_mlpackage: str,
    max_diff_threshold: float = 0.1,
):
    """Compare CoreML pipeline output against PyTorch reference on test audio."""
    print("\n[4/4] Validating conversion...")

    # 2-second test signal
    n_samples = SAMPLE_RATE * 2
    test_audio = torch.randn(1, n_samples) * 0.1  # small amplitude to avoid clipping
    test_length = torch.tensor([n_samples], dtype=torch.int32)

    # --- PyTorch reference (raw logits, bypassing log_softmax to match CoreML) ---
    model.eval()
    padded = torch.nn.functional.pad(test_audio, (0, MAX_AUDIO_SAMPLES - n_samples))
    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=padded, length=torch.tensor([MAX_AUDIO_SAMPLES]))
        encoded, _ = model.encoder(audio_signal=mel, length=mel_len.to(torch.int32))
        # Access decoder_layers directly to get raw logits (skip log_softmax)
        raw_logits = model.decoder.decoder_layers(encoded)  # [B, V, T]
        logits_pt = raw_logits.transpose(1, 2)  # [B, T, V]

    pt_output = logits_pt.float().numpy()
    print(f"  PyTorch shape: {pt_output.shape}  range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")

    # --- CoreML pipeline ---
    mel_model = ct.models.MLModel(mel_mlpackage)
    enc_model = ct.models.MLModel(encoder_mlpackage)

    audio_np = padded.numpy().astype(np.float32)
    length_np = np.array([MAX_AUDIO_SAMPLES], dtype=np.int32)

    mel_result = mel_model.predict({"audio": audio_np, "audio_length": length_np})
    mel_feat = mel_result.get("melspectrogram_features")
    mel_length_val = mel_result.get("mel_length")

    if mel_feat is None:
        print(f"  ERROR: melspectrogram_features not in mel model output: {list(mel_result.keys())}")
        return False

    if mel_length_val is not None:
        ml_len_np = mel_length_val if isinstance(mel_length_val, np.ndarray) else np.array([mel_length_val])
    else:
        ml_len_np = np.array([mel_feat.shape[-1]], dtype=np.int32)

    enc_result = enc_model.predict({
        "melspectrogram_features": mel_feat.astype(np.float32),
        "mel_length": ml_len_np.astype(np.int32),
    })

    coreml_out = enc_result.get("ctc_head_raw_output")
    if coreml_out is None:
        print(f"  ERROR: ctc_head_raw_output not in encoder output: {list(enc_result.keys())}")
        return False

    print(f"  CoreML shape:  {coreml_out.shape}  range: [{coreml_out.min():.4f}, {coreml_out.max():.4f}]")

    # Compare overlapping time steps
    min_t = min(pt_output.shape[1], coreml_out.shape[1])
    diff = np.abs(pt_output[0, :min_t, :] - coreml_out[0, :min_t, :]).max()
    print(f"  Max absolute difference: {diff:.6f}  (threshold: {max_diff_threshold})")

    # Argmax agreement check (what actually matters for CTC decoding)
    pt_argmax = np.argmax(pt_output[0, :min_t, :], axis=-1)
    cm_argmax = np.argmax(coreml_out[0, :min_t, :], axis=-1)
    argmax_agree = np.sum(pt_argmax == cm_argmax)
    argmax_total = min_t
    print(f"  Argmax agreement: {argmax_agree}/{argmax_total} frames ({100*argmax_agree/argmax_total:.1f}%)")

    if diff < max_diff_threshold:
        print("  ✓ Validation PASSED (absolute diff)")
        return True
    elif argmax_agree == argmax_total:
        print(f"  ✓ Validation PASSED (100% argmax agreement, abs diff {diff:.2f} above threshold — fp16 tail noise)")
        return True
    elif argmax_agree / argmax_total > 0.99:
        print(f"  ~ Validation MARGINAL ({100*argmax_agree/argmax_total:.1f}% argmax agreement)")
        return True
    else:
        print("  ✗ Validation FAILED — argmax divergence detected")
        return False


# ---------------------------------------------------------------------------
# Config export
# ---------------------------------------------------------------------------

def export_config(model, model_path: str, output_dir: Path):
    config = {
        "model_name": model_path,
        "architecture": "FastConformer-CTC",
        "sample_rate": SAMPLE_RATE,
        "max_audio_samples": MAX_AUDIO_SAMPLES,
        "vocab_size_with_blank": model.decoder.num_classes_with_blank,
        "output_format": "raw_logits",
        "ctc_output_name": "ctc_head_raw_output",
        "notes": "Convert .mlpackage → .mlmodelc with: xcrun coremlcompiler compile *.mlpackage <output_dir>",
    }
    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo CTC Vietnamese model to CoreML for FluidAudio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-ctc-0.6b-Vietnamese",
        help="HuggingFace model name or /path/to/model.nemo",
    )
    parser.add_argument(
        "--output",
        default="./parakeet-ctc-0.6b-vietnamese-coreml",
        help="Output directory (default: ./parakeet-ctc-0.6b-vietnamese-coreml)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip CoreML vs PyTorch validation (faster, but risky)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NeMo model
    model = load_nemo_model(args.model)

    mel_pkg = str(output_dir / "MelSpectrogram.mlpackage")
    enc_pkg = str(output_dir / "AudioEncoder.mlpackage")

    # Convert MelSpectrogram
    convert_mel_spectrogram(model.preprocessor, mel_pkg)

    # Convert AudioEncoder + CTC head
    convert_audio_encoder(model.encoder, model.decoder, model.preprocessor, enc_pkg)

    # Export vocabulary
    export_vocabulary(model, output_dir)

    # Config
    export_config(model, args.model, output_dir)

    # Validate
    if not args.skip_validation:
        validate_conversion(model, mel_pkg, enc_pkg)

    print(f"\n{'='*60}")
    print(f"Done! Output directory: {output_dir}/")
    print(f"\nFiles ready for FluidAudio:")
    print(f"  vocab.json               ← copy as-is")
    print(f"  config.json              ← copy as-is")
    print(f"  MelSpectrogram.mlpackage ← compile with xcrun")
    print(f"  AudioEncoder.mlpackage   ← compile with xcrun")
    print(f"\nCompile to .mlmodelc on macOS:")
    print(f"  xcrun coremlcompiler compile {output_dir}/MelSpectrogram.mlpackage {output_dir}/")
    print(f"  xcrun coremlcompiler compile {output_dir}/AudioEncoder.mlpackage {output_dir}/")
    print(f"\nThen upload MelSpectrogram.mlmodelc, AudioEncoder.mlmodelc, vocab.json, config.json")
    print(f"to your HuggingFace repo: FluidInference/parakeet-ctc-0.6b-vietnamese-coreml")


if __name__ == "__main__":
    main()


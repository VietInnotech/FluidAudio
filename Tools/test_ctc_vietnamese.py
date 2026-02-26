#!/usr/bin/env python3
"""
End-to-end test of the converted CoreML CTC Vietnamese model.

Downloads a real Vietnamese audio sample from FLEURS, runs it through:
  audio → MelSpectrogram.mlmodelc → AudioEncoder.mlmodelc → CTC greedy decode → text

Usage:
    cd Tools && .venv/bin/python test_ctc_vietnamese.py
    .venv/bin/python test_ctc_vietnamese.py --audio /path/to/audio.wav
"""

import argparse
import json
import math
import sys
from pathlib import Path

import coremltools as ct
import numpy as np

# ---------------------------------------------------------------------------
# Constants (must match conversion script and FluidAudio's ASRConstants)
# ---------------------------------------------------------------------------
MAX_AUDIO_SAMPLES = 240_000   # 15s at 16kHz
SAMPLE_RATE = 16_000
MODEL_DIR = Path(__file__).parent / "parakeet-ctc-0.6b-vietnamese-coreml"


# ---------------------------------------------------------------------------
# Audio loading / resampling
# ---------------------------------------------------------------------------

def load_audio_16khz(path: str) -> np.ndarray:
    """Load audio file, resample to 16kHz mono float32 in [-1, 1]."""
    import torchaudio
    import torch

    waveform, sr = torchaudio.load(path)
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample to 16kHz
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    samples = waveform.squeeze(0).numpy().astype(np.float32)
    print(f"  Audio: {len(samples)/SAMPLE_RATE:.2f}s, {len(samples)} samples, "
          f"range [{samples.min():.3f}, {samples.max():.3f}]")
    return samples


def download_fleurs_sample() -> str:
    """Download a single Vietnamese audio file from mozilla CommonVoice on HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; install with: pip install datasets")
        sys.exit(1)

    print("Downloading Vietnamese sample from mozilla-foundation/common_voice_17_0...")
    # streaming=True avoids downloading the full dataset
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "vi",
        split="test",
        streaming=True,
        trust_remote_code=False,
    )
    sample = next(iter(ds))
    audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    transcript = sample.get("sentence", "")

    # Save to temp file so we can load it uniformly
    import tempfile, soundfile as sf
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio_array, sr)
    print(f"  Reference transcript: {transcript!r}")
    return tmp, transcript


# ---------------------------------------------------------------------------
# CTC greedy decoder (mirrors CtcGreedyDecoder.swift)
# ---------------------------------------------------------------------------

WORD_BOUNDARY = "\u2581"  # U+2581 "▁"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable log-softmax over last axis."""
    max_l = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_l
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def ctc_greedy_decode(log_probs: np.ndarray, vocab: dict[int, str], blank_id: int) -> str:
    """
    CTC greedy decode: argmax per frame → collapse consecutive → remove blanks → text.
    log_probs: [T, V] float32 (log probabilities)
    """
    token_ids = np.argmax(log_probs, axis=-1)  # [T]

    # Collapse consecutive identical tokens and remove blanks
    tokens = []
    prev = -1
    for tid in token_ids:
        if tid != prev:
            if tid != blank_id:
                tokens.append(int(tid))
            prev = tid

    # Assemble text from BPE pieces
    pieces = [vocab.get(t, f"<unk_{t}>") for t in tokens]
    text = ""
    for piece in pieces:
        if piece.startswith(WORD_BOUNDARY):
            text += " " + piece[1:]
        else:
            text += piece
    return text.strip()


# ---------------------------------------------------------------------------
# CoreML inference pipeline
# ---------------------------------------------------------------------------

def run_pipeline(audio_samples: np.ndarray, model_dir: Path, vocab: dict[int, str]) -> str:
    mel_path = str(model_dir / "MelSpectrogram.mlpackage")
    enc_path = str(model_dir / "AudioEncoder.mlpackage")

    vocab_size = len(vocab)   # 1024
    blank_id = vocab_size     # 1024 (index after last vocab token)

    print(f"\nLoading CoreML models from {model_dir}/")
    mel_model = ct.models.MLModel(mel_path)
    enc_model = ct.models.MLModel(enc_path)
    print("  Models loaded.")

    # --- Chunk if longer than MAX_AUDIO_SAMPLES -----------------------
    chunk_size = MAX_AUDIO_SAMPLES
    overlap_samples = SAMPLE_RATE * 2  # 2s overlap
    stride = chunk_size - overlap_samples

    n = len(audio_samples)
    if n <= chunk_size:
        chunks = [(0, n)]
    else:
        chunks = []
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append((start, end))
            if end >= n:
                break
            start += stride

    print(f"  Chunks: {len(chunks)} (audio {n/SAMPLE_RATE:.1f}s)")

    all_log_probs = []
    frame_duration = None

    for chunk_idx, (start, end) in enumerate(chunks):
        chunk = audio_samples[start:end]
        clamped = len(chunk)

        # Pad to MAX_AUDIO_SAMPLES
        padded = np.zeros(MAX_AUDIO_SAMPLES, dtype=np.float32)
        padded[:clamped] = chunk
        audio_np = padded.reshape(1, MAX_AUDIO_SAMPLES)
        length_np = np.array([MAX_AUDIO_SAMPLES], dtype=np.int32)

        # Mel spectrogram
        mel_result = mel_model.predict({
            "audio": audio_np,
            "audio_length": length_np,
        })
        mel_feat = mel_result["melspectrogram_features"].astype(np.float32)
        mel_len = mel_result.get("mel_length")
        if mel_len is not None:
            ml = int(np.array(mel_len).flatten()[0])
        else:
            ml = mel_feat.shape[-1]

        if chunk_idx == 0:
            print(f"  Mel shape: {mel_feat.shape}, mel_length: {ml}")

        # Encoder
        enc_result = enc_model.predict({
            "melspectrogram_features": mel_feat.astype(np.float32),
            "mel_length": np.array([ml], dtype=np.int32),
        })
        raw_logits = enc_result["ctc_head_raw_output"]  # [1, T, V]

        if chunk_idx == 0:
            print(f"  Encoder output shape: {raw_logits.shape}, "
                  f"range [{raw_logits.min():.2f}, {raw_logits.max():.2f}]")

        # [1, T, V] → [T, V]
        logits_2d = raw_logits.reshape(-1, raw_logits.shape[-1])

        # Compute log-softmax
        log_probs = log_softmax(logits_2d.astype(np.float64)).astype(np.float32)

        # Trim padding frames proportionally
        total_frames = log_probs.shape[0]
        active_frames = max(1, round(total_frames * clamped / MAX_AUDIO_SAMPLES))
        log_probs = log_probs[:active_frames]

        if frame_duration is None and active_frames > 0:
            frame_duration = clamped / active_frames / SAMPLE_RATE

        # Handle overlap between chunks
        if not all_log_probs:
            all_log_probs.extend(log_probs.tolist())
        else:
            overlap_frames = round(overlap_samples / SAMPLE_RATE / frame_duration) if frame_duration else 0
            overlap_count = min(overlap_frames, len(all_log_probs), len(log_probs))
            # Average overlapping frames
            for i in range(overlap_count):
                existing = np.array(all_log_probs[len(all_log_probs) - overlap_count + i])
                new_frame = log_probs[i]
                all_log_probs[len(all_log_probs) - overlap_count + i] = ((existing + new_frame) / 2).tolist()
            # Append non-overlapping
            all_log_probs.extend(log_probs[overlap_count:].tolist())

    print(f"  Total frames: {len(all_log_probs)}, frame_duration: {frame_duration:.4f}s")

    # CTC greedy decode
    log_probs_np = np.array(all_log_probs, dtype=np.float32)
    text = ctc_greedy_decode(log_probs_np, vocab, blank_id)
    return text


# ---------------------------------------------------------------------------
# Vocabulary loading
# ---------------------------------------------------------------------------

def load_vocab(model_dir: Path) -> dict[int, str]:
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test CTC Vietnamese CoreML pipeline on real audio")
    parser.add_argument("--audio", default=None, help="Path to audio file (WAV/FLAC/MP3). If omitted, downloads from FLEURS.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Directory with .mlmodelc files and vocab.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Run convert_nemo_ctc_to_coreml.py first to generate models.")
        sys.exit(1)

    # Load vocab
    vocab = load_vocab(model_dir)
    print(f"Vocabulary: {len(vocab)} tokens, blank_id={len(vocab)}")
    print(f"  Sample: {dict(list(vocab.items())[:6])}")

    # Get audio
    reference_transcript = None
    if args.audio:
        audio_path = args.audio
        print(f"\nLoading audio: {audio_path}")

    print("Loading audio...")
    audio_samples = load_audio_16khz(audio_path)

    # Run inference
    print("\nRunning CoreML inference pipeline...")
    transcript = run_pipeline(audio_samples, model_dir, vocab)

    print(f"\n{'='*60}")
    print(f"TRANSCRIPT: {transcript!r}")
    if reference_transcript:
        print(f"REFERENCE:  {reference_transcript!r}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

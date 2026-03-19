#!/usr/bin/env python3
"""Convert EraX-WoW-Turbo V1.1 to CoreML for FluidAudio.

Downloads erax-ai/EraX-WoW-Turbo-V1.1 (a fine-tuned whisper-large-v3-turbo) and
converts it to the same CoreML format used by WhisperKit / FluidAudio's WhisperManager.

The converted models are placed in:
    ./build/erax-wow-turbo-v1.1-coreml/

Output files (compatible with WhisperModels.load()):
    MelSpectrogram.mlmodelc/
    AudioEncoder.mlmodelc/
    TextDecoder.mlmodelc/
    TextDecoderContextPrefill.mlmodelc/   (may be absent for some models)
    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt
    normalizer.json
    special_tokens_map.json
    added_tokens.json

Usage:
    # First activate the tools virtual environment:
    cd /path/to/FluidAudio
    uv run Tools/convert_erax_whisper_to_coreml.py

    # Or with explicit output directory:
    uv run Tools/convert_erax_whisper_to_coreml.py --output-dir ./build/erax-wow-turbo-v1.1-coreml

Requirements (added to pyproject.toml or installed manually):
    pip install 'whisperkittools @ git+https://github.com/argmaxinc/whisperkittools.git'
    pip install huggingface_hub transformers

Notes:
    - The model shares the exact same architecture as whisper-large-v3-turbo
      (d_model=1280, 4 decoder layers, vocab_size=51866) so the generated CoreML
      models are drop-in replacements for the argmaxinc/whisperkit-coreml models.
    - The tokenizer is identical to openai/whisper-large-v3-turbo; we download it
      from that repo to guarantee compatibility with WhisperKit's tokenizer loader.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HF_MODEL_ID = "erax-ai/EraX-WoW-Turbo-V1.1"
HF_TOKENIZER_REPO = "openai/whisper-large-v3-turbo"
DEFAULT_OUTPUT_DIR = Path("./build/erax-wow-turbo-v1.1-coreml")

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
]

REQUIRED_MODELS = [
    "MelSpectrogram.mlmodelc",
    "AudioEncoder.mlmodelc",
    "TextDecoder.mlmodelc",
]


def run(cmd: list[str], **kwargs) -> None:
    """Run a command and raise on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def check_whisperkit_tools() -> None:
    """Verify whisperkit is importable; print install hint if not."""
    try:
        import whisperkit  # noqa: F401
    except ImportError:
        sys.exit(
            "whisperkit not found. Install with:\n"
            "  pip install 'whisperkit @ git+https://github.com/argmaxinc/whisperkittools.git'\n"
            "or add it to your pyproject.toml dependencies."
        )


def convert_model(output_dir: Path) -> None:
    """Use whisperkit-generate-model to convert HuggingFace weights to CoreML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Converting {HF_MODEL_ID} to CoreML...")
    print(f"      Output dir: {output_dir.resolve()}\n")

    # whiskerkit-generate-model writes directly to output_dir.
    # --generate-decoder-context-prefill-data pre-computes KV cache for faster decoding.
    # --disable-default-tests skips expensive testing (helps with disk space constraints).
    run([
        "whisperkit-generate-model",
        "--model-version", HF_MODEL_ID,
        "--output-dir", str(output_dir),
        "--generate-decoder-context-prefill-data",
        "--disable-default-tests",
    ])

    # whisperkit-generate-model creates a subdirectory named after the model.
    # Move the CoreML models to the root of output_dir.
    model_subdir = output_dir / HF_MODEL_ID.replace("/", "_")
    if model_subdir.exists():
        print(f"\n  Moving models from {model_subdir.name}/ to root...")
        for item in model_subdir.iterdir():
            if item.is_dir() and (item.name.endswith(".mlmodelc") or item.name.endswith(".mlpackage")):
                dest = output_dir / item.name
                if dest.exists():
                    import shutil
                    shutil.rmtree(dest)
                item.rename(dest)
                print(f"    → {item.name}")
        # Remove empty subdirectory
        try:
            model_subdir.rmdir()
        except OSError:
            pass  # Directory not empty, leave it


def ensure_tokenizer_files(output_dir: Path) -> None:
    """Download tokenizer files from openai/whisper-large-v3-turbo if missing.

    whisperkit-generate-model may not always produce all tokenizer files expected by
    swift-transformers (AutoTokenizer.from(modelFolder:)). We supplement from the
    canonical Whisper tokenizer repo.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [warning] huggingface_hub not found; skipping tokenizer file check.")
        return

    print(f"\n[2/3] Verifying tokenizer files from {HF_TOKENIZER_REPO}...")
    for filename in TOKENIZER_FILES:
        dest = output_dir / filename
        if dest.exists():
            print(f"  ✓ {filename} (already present)")
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=HF_TOKENIZER_REPO,
                filename=filename,
                local_dir=str(output_dir),
            )
            print(f"  ↓ {filename}  →  {downloaded}")
        except Exception as exc:
            print(f"  [warning] Could not download {filename}: {exc}")


def verify_output(output_dir: Path) -> bool:
    """Check that the required CoreML bundles and tokenizer.json are present."""
    print(f"\n[3/3] Verifying output in {output_dir}...")
    ok = True
    for name in REQUIRED_MODELS + ["tokenizer.json"]:
        path = output_dir / name
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if not exists:
            ok = False

    # TextDecoderContextPrefill is optional
    prefill = output_dir / "TextDecoderContextPrefill.mlmodelc"
    print(f"  {'✓' if prefill.exists() else '○'} TextDecoderContextPrefill.mlmodelc  (optional)")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EraX-WoW-Turbo V1.1 to CoreML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CoreML models (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    check_whisperkit_tools()
    convert_model(args.output_dir)
    ensure_tokenizer_files(args.output_dir)

    if not verify_output(args.output_dir):
        sys.exit("\nConversion incomplete — some required files are missing.")

    print(f"\n✅ Conversion complete: {args.output_dir.resolve()}")
    print("\nNext steps:")
    print("  1. Upload to HuggingFace:")
    print("     bash Tools/upload_erax_whisper.sh")
    print("  2. Build FluidAudio:")
    print("     swift build")


if __name__ == "__main__":
    main()

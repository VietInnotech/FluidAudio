#!/usr/bin/env python3
"""
INT4 linear quantization for VibeVoice-ASR CoreML models.

Applies per-block INT4 symmetric quantization to the decoder (13 GB → ~3.7 GB).
Encoders are also quantized (optional, smaller benefit since they're mostly conv ops).
Embeddings and vocab are copied unchanged (already float16, sparsely accessed).

Usage:
    uv run python quantize_vibevoice_int4.py [--decoder-only] [--skip-compile]
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto


BASE_DIR = Path(__file__).parent.parent
F32_DIR = BASE_DIR / "build" / "vibevoice-asr-coreml" / "f32"
INT4_DIR = BASE_DIR / "build" / "vibevoice-asr-coreml" / "int4"
COMPILED_INT4_DIR = BASE_DIR / "Models" / "vibevoice-asr-coreml" / "int4"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def size_gb(path: Path) -> str:
    result = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    return result.stdout.split()[0] if result.returncode == 0 else "?"


def make_int4_config(weight_threshold: int = 512) -> cto.coreml.OptimizationConfig:
    """INT4 per-block symmetric quantization — standard for LLM decoders."""
    op_config = cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
        weight_threshold=weight_threshold,
    )
    return cto.coreml.OptimizationConfig(global_config=op_config)


def make_int8_config(weight_threshold: int = 512) -> cto.coreml.OptimizationConfig:
    """INT8 per-channel quantization for encoders (mostly conv ops)."""
    op_config = cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
        weight_threshold=weight_threshold,
    )
    return cto.coreml.OptimizationConfig(global_config=op_config)


def quantize_model(
    name: str,
    src: Path,
    dst: Path,
    config: cto.coreml.OptimizationConfig,
) -> None:
    log(f"Loading {name} ({size_gb(src)}) ...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(str(src))

    log(f"Quantizing {name} ...")
    compressed = cto.coreml.linear_quantize_weights(mlmodel, config)

    log(f"Saving to {dst} ...")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    compressed.save(str(dst))

    elapsed = time.time() - t0
    log(f"Done: {name}  {size_gb(src)} → {size_gb(dst)}  ({elapsed:.0f}s)")


def compile_model(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + ".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    log(f"Compiling {src.name} → {dst.name} ...")
    t0 = time.time()
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(src), str(out_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"coremlcompiler failed for {src.name}")
    elapsed = time.time() - t0
    log(f"Compiled {dst.name} ({size_gb(dst)}) in {elapsed:.0f}s")
    return dst


def copy_unchanged(name: str, src: Path, dst_dir: Path) -> None:
    dst = dst_dir / name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    log(f"Copied {name} → {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="INT4 quantize VibeVoice-ASR CoreML models")
    parser.add_argument("--decoder-only", action="store_true", help="Only quantize the decoder (skip encoders)")
    parser.add_argument("--skip-compile", action="store_true", help="Skip xcrun coremlcompiler step")
    args = parser.parse_args()

    log("=" * 60)
    log("VibeVoice-ASR INT4 Quantization")
    log("=" * 60)
    log(f"Source : {F32_DIR}")
    log(f"Output : {INT4_DIR}")
    log(f"Compiled: {COMPILED_INT4_DIR}")

    if not F32_DIR.exists():
        log(f"ERROR: Source directory not found: {F32_DIR}")
        sys.exit(1)

    INT4_DIR.mkdir(parents=True, exist_ok=True)
    COMPILED_INT4_DIR.mkdir(parents=True, exist_ok=True)

    # ── Decoder: INT4 per-block (the big one) ─────────────────────────────────
    decoder_src = F32_DIR / "vibevoice_decoder_stateful.mlpackage"
    decoder_dst = INT4_DIR / "vibevoice_decoder_stateful.mlpackage"
    log(f"\nStep 1/4 — Decoder INT4 quantization")
    quantize_model("vibevoice_decoder_stateful", decoder_src, decoder_dst, make_int4_config())

    if not args.decoder_only:
        # ── Acoustic encoder: INT8 per-channel ────────────────────────────────
        acoustic_src = F32_DIR / "vibevoice_acoustic_encoder.mlpackage"
        acoustic_dst = INT4_DIR / "vibevoice_acoustic_encoder.mlpackage"
        log(f"\nStep 2/4 — Acoustic encoder INT8 quantization")
        quantize_model("vibevoice_acoustic_encoder", acoustic_src, acoustic_dst, make_int8_config())

        # ── Semantic encoder: INT8 per-channel ────────────────────────────────
        semantic_src = F32_DIR / "vibevoice_semantic_encoder.mlpackage"
        semantic_dst = INT4_DIR / "vibevoice_semantic_encoder.mlpackage"
        log(f"\nStep 3/4 — Semantic encoder INT8 quantization")
        quantize_model("vibevoice_semantic_encoder", semantic_src, semantic_dst, make_int8_config())
    else:
        log("\nStep 2-3/4 — Encoders: copying unchanged (--decoder-only flag set)")
        shutil.copytree(F32_DIR / "vibevoice_acoustic_encoder.mlpackage",
                        INT4_DIR / "vibevoice_acoustic_encoder.mlpackage",
                        dirs_exist_ok=True)
        shutil.copytree(F32_DIR / "vibevoice_semantic_encoder.mlpackage",
                        INT4_DIR / "vibevoice_semantic_encoder.mlpackage",
                        dirs_exist_ok=True)

    # ── Static assets: copy unchanged ─────────────────────────────────────────
    log(f"\nStep 4/4 — Copying static assets (embeddings, vocab, metadata)")
    for name in ["vibevoice_embeddings.bin", "vocab.json", "metadata.json"]:
        copy_unchanged(name, F32_DIR / name, INT4_DIR)

    # ── Compile all mlpackages ─────────────────────────────────────────────────
    if not args.skip_compile:
        log("\nCompiling all mlpackages → mlmodelc ...")
        for pkg in INT4_DIR.glob("*.mlpackage"):
            compile_model(pkg, COMPILED_INT4_DIR)

        # Copy flat files to compiled dir too
        for name in ["vibevoice_embeddings.bin", "vocab.json", "metadata.json"]:
            copy_unchanged(name, INT4_DIR / name, COMPILED_INT4_DIR)

    # ── Final summary ──────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("INT4 Quantization Complete")
    log("=" * 60)
    log(f"Packages : {size_gb(INT4_DIR)}")
    if not args.skip_compile:
        log(f"Compiled : {size_gb(COMPILED_INT4_DIR)}")

    log("\nFiles in int4 package dir:")
    for f in sorted(INT4_DIR.iterdir()):
        log(f"  {f.name}  ({size_gb(f)})")

    if not args.skip_compile:
        log("\nFiles in int4 compiled dir:")
        for f in sorted(COMPILED_INT4_DIR.iterdir()):
            log(f"  {f.name}  ({size_gb(f)})")


if __name__ == "__main__":
    main()

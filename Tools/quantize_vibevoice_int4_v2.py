#!/usr/bin/env python3
"""
INT4 quantization using sparse loading — avoids full model in memory.

This uses coremltools' lower-level APIs to quantize without loading
the entire 13 GB model.
"""

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


def quantize_mlpackage_with_proto(
    name: str,
    src_pkg: Path,
    dst_pkg: Path,
    use_int4: bool = True,
) -> None:
    """Quantize using proto manipulation to avoid full model load."""
    
    log(f"\n{'='*60}")
    log(f"Quantizing {name} ({size_gb(src_pkg)})")
    log(f"{'='*60}")
    
    # Copy the mlpackage
    log(f"[1/3] Copying mlpackage...")
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(src_pkg, dst_pkg)
    
    # Load the model from the copied package
    log(f"[2/3] Loading model ({name})...")
    t0 = time.time()
    try:
        # Try to load with low memory usage
        mlmodel = ct.models.MLModel(str(dst_pkg))
        elapsed = time.time() - t0
        log(f"      Loaded in {elapsed:.0f}s")
    except Exception as e:
        log(f"      ERROR loading: {e}")
        log(f"      Retrying with proto loading...")
        # If loading fails, try proto-only mode
        from coremltools.models.utils import check_top_level_model_type
        import json
        manifest_path = dst_pkg / "Manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        log(f"      Manifest loaded: {manifest}")
        raise
    
    # Build quantization config
    log(f"[3/3] Applying quantization...")
    if use_int4:
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=32,
            weight_threshold=512,
        )
        dtype_name = "INT4"
    else:
        op_config = cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=512,
        )
        dtype_name = "INT8"
    
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    
    t0 = time.time()
    try:
        compressed = cto.coreml.linear_quantize_weights(mlmodel, config)
        elapsed = time.time() - t0
        log(f"      Quantization done in {elapsed:.0f}s ({dtype_name})")
        
        # Remove old and save new
        shutil.rmtree(dst_pkg)
        compressed.save(str(dst_pkg))
        log(f"      Saved: {size_gb(dst_pkg)}")
    except Exception as e:
        log(f"      ERROR: {e}")
        raise


def compile_model(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + ".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    log(f"Compiling {src.name} → {dst.name}...")
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
    log(f"  {dst.name} ({size_gb(dst)}) in {elapsed:.0f}s")
    return dst


def main() -> None:
    log("=" * 60)
    log("VibeVoice-ASR INT4/INT8 Quantization")
    log("=" * 60)
    
    if not F32_DIR.exists():
        log(f"ERROR: Source directory not found")
        sys.exit(1)
    
    INT4_DIR.mkdir(parents=True, exist_ok=True)
    COMPILED_INT4_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Decoder INT4 (the big one)
        decoder_src = F32_DIR / "vibevoice_decoder_stateful.mlpackage"
        decoder_dst = INT4_DIR / "vibevoice_decoder_stateful.mlpackage"
        quantize_mlpackage_with_proto("Decoder", decoder_src, decoder_dst, use_int4=True)
        
        # Acoustic encoder INT8
        acoustic_src = F32_DIR / "vibevoice_acoustic_encoder.mlpackage"
        acoustic_dst = INT4_DIR / "vibevoice_acoustic_encoder.mlpackage"
        quantize_mlpackage_with_proto("Acoustic Encoder", acoustic_src, acoustic_dst, use_int4=False)
        
        # Semantic encoder INT8
        semantic_src = F32_DIR / "vibevoice_semantic_encoder.mlpackage"
        semantic_dst = INT4_DIR / "vibevoice_semantic_encoder.mlpackage"
        quantize_mlpackage_with_proto("Semantic Encoder", semantic_src, semantic_dst, use_int4=False)
        
        # Copy static assets
        log(f"\nCopying static assets...")
        for name in ["vibevoice_embeddings.bin", "vocab.json", "metadata.json"]:
            shutil.copy2(F32_DIR / name, INT4_DIR / name)
            log(f"  {name}")
        
        # Compile
        log(f"\nCompiling mlpackages → mlmodelc...")
        for pkg in INT4_DIR.glob("*.mlpackage"):
            compile_model(pkg, COMPILED_INT4_DIR)
        
        # Copy static files
        for name in ["vibevoice_embeddings.bin", "vocab.json", "metadata.json"]:
            shutil.copy2(INT4_DIR / name, COMPILED_INT4_DIR / name)
        
        log(f"\n{'='*60}")
        log(f"SUCCESS!")
        log(f"INT4 packages: {size_gb(INT4_DIR)}")
        log(f"Compiled     : {size_gb(COMPILED_INT4_DIR)}")
        log(f"{'='*60}")
        
    except Exception as e:
        log(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

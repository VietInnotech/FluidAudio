#!/usr/bin/env python3
"""
FluidAudio Benchmark Suite

Modes:
1) Built-in benchmarks (LibriSpeech / VAD / AMI)
2) Hugging Face custom ASR benchmark mode via --hf-dataset

Recommended invocation (uses uv-managed environment from `Tools/pyproject.toml`):
  uv run --project Tools python run_benchmarks.py --hf-dataset doof-ferb/fpt_fosd
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Baseline values from Documentation/Benchmarks.md
BASELINES = {
    "asr": {
        "wer_percent": 5.8,
        "rtfx_min": 200,  # M4 Pro: ~210x
        "description": "LibriSpeech test-clean, Parakeet TDT 0.6B",
    },
    "vad": {
        "f1_percent": 85.0,
        "rtfx_min": 500,
        "description": "VOiCES dataset, Silero VAD",
    },
    "diarization": {
        "der_percent": 17.7,
        "rtfx_min": 1.0,
        "description": "AMI SDM, pyannote-based",
    },
}

HF_MODEL_CHOICES = {"asr-v2", "asr-v3", "qwen3-f32", "qwen3-int8"}


def run_command(cmd: list[str], output_file: Path | None = None) -> tuple[int, str]:
    """Run a command and optionally save output."""
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    if output_file:
        output_file.write_text(output)

    return result.returncode, output


def build_release() -> bool:
    """Build the project in release mode."""
    print("\n" + "=" * 60)
    print("Building release...")
    print("=" * 60)

    returncode, _ = run_command(["swift", "build", "-c", "release"])

    if returncode != 0:
        print("ERROR: Build failed!")
        return False

    print("Build successful.")
    return True


def run_asr_benchmark(output_dir: Path, quick: bool = False) -> dict | None:
    """Run ASR benchmark on LibriSpeech test-clean."""
    print("\n" + "=" * 60)
    print("ASR Benchmark (LibriSpeech test-clean)")
    print("=" * 60)

    max_files = "100" if quick else "all"
    output_json = output_dir / "asr_results.json"

    cmd = [
        "swift",
        "run",
        "-c",
        "release",
        "fluidaudiocli",
        "asr-benchmark",
        "--subset",
        "test-clean",
        "--max-files",
        max_files,
        "--output",
        str(output_json),
    ]

    returncode, _ = run_command(cmd, output_dir / "asr_log.txt")

    if returncode != 0:
        print("ERROR: ASR benchmark failed!")
        return None

    if output_json.exists():
        return json.loads(output_json.read_text())

    return None


def run_vad_benchmark(output_dir: Path, quick: bool = False) -> dict | None:
    """Run VAD benchmark."""
    print("\n" + "=" * 60)
    print("VAD Benchmark")
    print("=" * 60)

    dataset = "mini50" if quick else "voices-subset"
    output_json = output_dir / "vad_results.json"

    cmd = [
        "swift",
        "run",
        "-c",
        "release",
        "fluidaudiocli",
        "vad-benchmark",
        "--dataset",
        dataset,
        "--all-files",
        "--threshold",
        "0.5",
        "--output",
        str(output_json),
    ]

    returncode, _ = run_command(cmd, output_dir / "vad_log.txt")

    if returncode != 0:
        print("ERROR: VAD benchmark failed!")
        return None

    if output_json.exists():
        return json.loads(output_json.read_text())

    return None


def run_diarization_benchmark(output_dir: Path, quick: bool = False) -> dict | None:
    """Run diarization benchmark on AMI SDM."""
    print("\n" + "=" * 60)
    print("Diarization Benchmark (AMI SDM)")
    print("=" * 60)

    output_json = output_dir / "diarization_results.json"

    cmd = [
        "swift",
        "run",
        "-c",
        "release",
        "fluidaudiocli",
        "diarization-benchmark",
        "--auto-download",
        "--output",
        str(output_json),
    ]

    if quick:
        cmd.extend(["--single-file", "ES2004a"])

    returncode, _ = run_command(cmd, output_dir / "diarization_log.txt")

    if returncode != 0:
        print("ERROR: Diarization benchmark failed!")
        return None

    if output_json.exists():
        return json.loads(output_json.read_text())

    return None


def compare_results(results: dict) -> None:
    """Compare results against built-in baselines."""
    print("\n" + "=" * 60)
    print("Results vs Baselines (Documentation/Benchmarks.md)")
    print("=" * 60)

    if "asr" in results and results["asr"]:
        asr = results["asr"]
        baseline = BASELINES["asr"]
        summary = asr.get("summary", {})
        wer = summary.get("averageWER", asr.get("wer", asr.get("average_wer", 0))) * 100
        rtfx = summary.get("medianRTFx", asr.get("rtfx", asr.get("median_rtfx", 0)))

        wer_status = "✓" if wer <= baseline["wer_percent"] * 1.1 else "✗"
        rtfx_status = "✓" if rtfx >= baseline["rtfx_min"] * 0.8 else "✗"

        print(f"\nASR ({baseline['description']}):")
        print(f"  WER:  {wer:.1f}% (baseline: {baseline['wer_percent']}%) {wer_status}")
        print(f"  RTFx: {rtfx:.1f}x (baseline: {baseline['rtfx_min']}x+) {rtfx_status}")

    if "vad" in results and results["vad"]:
        vad = results["vad"]
        baseline = BASELINES["vad"]
        f1 = vad.get("f1_score", 0)
        rtfx = vad.get("rtfx", 0)

        f1_status = "✓" if f1 >= baseline["f1_percent"] * 0.9 else "✗"
        rtfx_status = "✓" if rtfx >= baseline["rtfx_min"] * 0.5 else "✗"

        print(f"\nVAD ({baseline['description']}):")
        print(f"  F1:   {f1:.1f}% (baseline: {baseline['f1_percent']}%+) {f1_status}")
        print(f"  RTFx: {rtfx:.1f}x (baseline: {baseline['rtfx_min']}x+) {rtfx_status}")

    if "diarization" in results and results["diarization"]:
        diar = results["diarization"]
        baseline = BASELINES["diarization"]
        der = diar.get("der", diar.get("average_der", 0)) * 100
        rtfx = diar.get("rtfx", diar.get("average_rtfx", 0))

        der_status = "✓" if der <= baseline["der_percent"] * 1.2 else "✗"
        rtfx_status = "✓" if rtfx >= baseline["rtfx_min"] else "✗"

        print(f"\nDiarization ({baseline['description']}):")
        print(f"  DER:  {der:.1f}% (baseline: {baseline['der_percent']}%) {der_status}")
        print(f"  RTFx: {rtfx:.1f}x (baseline: {baseline['rtfx_min']}x+) {rtfx_status}")


def parse_max_samples(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"all", "-1"}:
        return None

    parsed = int(lowered)
    if parsed <= 0:
        raise ValueError("--hf-max-samples must be positive or 'all'")
    return parsed


def parse_model_list(raw_models: str) -> list[str]:
    if raw_models.strip().lower() == "all":
        return sorted(HF_MODEL_CHOICES)

    models = [item.strip().lower() for item in raw_models.split(",") if item.strip()]
    if not models:
        raise ValueError("No models selected. Use --models with at least one model.")

    unknown = sorted(set(models) - HF_MODEL_CHOICES)
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Supported: {', '.join(sorted(HF_MODEL_CHOICES))}"
        )

    # Deduplicate while preserving order.
    unique_models: list[str] = []
    for model in models:
        if model not in unique_models:
            unique_models.append(model)
    return unique_models


def resolve_hf_columns(dataset, requested_audio: str | None, requested_text: str | None) -> tuple[str, str]:
    from datasets import Audio

    feature_map = dataset.features

    if requested_audio:
        if requested_audio not in feature_map:
            raise ValueError(f"Audio column '{requested_audio}' not found in dataset features.")
        audio_column = requested_audio
    else:
        audio_candidates = [name for name, feature in feature_map.items() if isinstance(feature, Audio)]
        if not audio_candidates:
            raise ValueError("Could not auto-detect audio column. Pass --hf-audio-column explicitly.")
        audio_column = audio_candidates[0]

    if requested_text:
        if requested_text not in feature_map:
            raise ValueError(f"Text column '{requested_text}' not found in dataset features.")
        text_column = requested_text
    else:
        text_candidates = ["transcription", "transcript", "text", "sentence", "normalized_text"]
        text_column = next((name for name in text_candidates if name in feature_map), "")
        if not text_column:
            raise ValueError("Could not auto-detect text column. Pass --hf-text-column explicitly.")

    return audio_column, text_column


def materialize_hf_dataset(
    dataset_name: str,
    config_name: str | None,
    split_name: str,
    output_dir: Path,
    max_samples: int | None,
    shuffle: bool,
    seed: int,
    audio_column: str | None,
    text_column: str | None,
) -> tuple[Path, dict]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HF dataset mode requires python package: datasets.\n"
            "Install with uv: uv add --project Tools datasets\n"
            "Run with uv: uv run --project Tools python run_benchmarks.py ..."
        ) from exc

    print("\n" + "=" * 60)
    print("Preparing Hugging Face Dataset")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Config: {config_name or 'default'}")
    print(f"Split: {split_name}")

    load_kwargs = {
        "split": split_name,
        "trust_remote_code": False,
    }
    if config_name:
        dataset = load_dataset(dataset_name, config_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)

    detected_audio, detected_text = resolve_hf_columns(dataset, audio_column, text_column)
    print(f"Audio column: {detected_audio}")
    print(f"Text column: {detected_text}")

    dataset = dataset.cast_column(detected_audio, Audio(decode=False))

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if max_samples is not None:
        selected = min(max_samples, len(dataset))
        dataset = dataset.select(range(selected))

    materialized_audio_dir = output_dir / "audio"
    materialized_audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    kept = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row_index, sample in enumerate(dataset):
            transcript_raw = sample.get(detected_text)
            transcript = str(transcript_raw).strip() if transcript_raw is not None else ""
            if not transcript:
                skipped += 1
                continue

            audio_info = sample.get(detected_audio)
            if not isinstance(audio_info, dict):
                skipped += 1
                continue

            source_name = str(audio_info.get("path") or "")
            source_suffix = Path(source_name).suffix.lower() if source_name else ""
            suffix = source_suffix if source_suffix else ".audio"
            audio_path = materialized_audio_dir / f"sample_{kept:06d}{suffix}"

            raw_bytes = audio_info.get("bytes")
            if isinstance(raw_bytes, (bytes, bytearray)):
                audio_path.write_bytes(raw_bytes)
            elif source_name:
                source_path = Path(source_name).expanduser()
                if not source_path.exists():
                    skipped += 1
                    continue
                shutil.copyfile(source_path, audio_path)
            else:
                skipped += 1
                continue

            manifest_entry = {
                "id": sample.get("id", row_index),
                "file_name": audio_path.name,
                "audio_path": str(audio_path.resolve()),
                "transcript": transcript,
            }
            manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            kept += 1

    if kept == 0:
        raise RuntimeError("No valid rows were materialized from the dataset.")

    metadata = {
        "dataset": dataset_name,
        "config": config_name or "default",
        "split": split_name,
        "audioColumn": detected_audio,
        "textColumn": detected_text,
        "rowsRequested": "all" if max_samples is None else max_samples,
        "rowsMaterialized": kept,
        "rowsSkipped": skipped,
        "totalAudioSeconds": None,
        "manifestPath": str(manifest_path),
    }

    print(f"Materialized rows: {kept} (skipped: {skipped})")
    print(f"Manifest: {manifest_path}")
    return manifest_path, metadata


def run_manifest_asr_benchmark(
    output_dir: Path, manifest_path: Path, dataset_label: str, model_version: str
) -> dict | None:
    output_json = output_dir / f"asr_{model_version}_results.json"
    log_file = output_dir / f"asr_{model_version}_log.txt"

    cmd = [
        "swift",
        "run",
        "-c",
        "release",
        "fluidaudiocli",
        "asr-benchmark",
        "--manifest",
        str(manifest_path),
        "--dataset-name",
        dataset_label,
        "--model-version",
        model_version,
        "--output",
        str(output_json),
    ]
    returncode, _ = run_command(cmd, log_file)
    if returncode != 0:
        print(f"ERROR: ASR {model_version} benchmark failed.")
        return None
    if not output_json.exists():
        print(f"ERROR: Expected output not found: {output_json}")
        return None
    return json.loads(output_json.read_text())


def run_manifest_qwen3_benchmark(
    output_dir: Path,
    manifest_path: Path,
    dataset_label: str,
    variant: str,
    language: str,
) -> dict | None:
    output_json = output_dir / f"qwen3_{variant}_results.json"
    log_file = output_dir / f"qwen3_{variant}_log.txt"

    cmd = [
        "swift",
        "run",
        "-c",
        "release",
        "fluidaudiocli",
        "qwen3-benchmark",
        "--manifest",
        str(manifest_path),
        "--dataset-name",
        dataset_label,
        "--variant",
        variant,
        "--output",
        str(output_json),
    ]
    if language:
        cmd.extend(["--language", language])

    returncode, _ = run_command(cmd, log_file)
    if returncode != 0:
        print(f"ERROR: Qwen3 {variant} benchmark failed.")
        return None
    if not output_json.exists():
        print(f"ERROR: Expected output not found: {output_json}")
        return None
    return json.loads(output_json.read_text())


def summarize_hf_results(results: dict) -> None:
    print("\n" + "=" * 60)
    print("HF Dataset Benchmark Summary")
    print("=" * 60)

    for model_name, model_data in results.items():
        if not model_data:
            print(f"- {model_name}: FAILED")
            continue

        summary = model_data.get("summary", {})
        avg_wer = summary.get("averageWER", 0.0) * 100.0
        avg_cer = summary.get("averageCER", 0.0) * 100.0
        overall_rtfx = summary.get("overallRTFx", 0.0)
        files_processed = summary.get("filesProcessed", 0)
        print(
            f"- {model_name}: files={files_processed}, WER={avg_wer:.1f}%, "
            f"CER={avg_cer:.1f}%, RTFx={overall_rtfx:.2f}x"
        )


def run_hf_benchmark_mode(args: argparse.Namespace, output_dir: Path, timestamp: str) -> int:
    models = parse_model_list(args.models)
    max_samples = parse_max_samples(args.hf_max_samples)

    print("=" * 60)
    print("FluidAudio HF Dataset Benchmark")
    print("=" * 60)
    print(f"Dataset: {args.hf_dataset}")
    print(f"Split: {args.hf_split}")
    print(f"Models: {', '.join(models)}")
    print(f"Max samples: {args.hf_max_samples}")
    print(f"Output: {output_dir}")
    print(f"Time: {timestamp}")

    if not args.no_build and not build_release():
        return 1

    dataset_dir = output_dir / "hf_materialized"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    manifest_path, metadata = materialize_hf_dataset(
        dataset_name=args.hf_dataset,
        config_name=args.hf_config,
        split_name=args.hf_split,
        output_dir=dataset_dir,
        max_samples=max_samples,
        shuffle=args.hf_shuffle,
        seed=args.hf_seed,
        audio_column=args.hf_audio_column,
        text_column=args.hf_text_column,
    )

    if args.prepare_only:
        prepared_output = output_dir / "hf_prepared.json"
        prepared_output.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "mode": "hf-prepare-only",
                    "dataset": metadata,
                    "manifestPath": str(manifest_path),
                },
                indent=2,
            )
        )
        print(f"\nPrepared dataset only. Metadata saved to: {prepared_output}")
        return 0

    hf_results: dict[str, dict | None] = {}
    dataset_label = args.hf_dataset.replace("/", "_")

    for model in models:
        print("\n" + "=" * 60)
        print(f"Running model: {model}")
        print("=" * 60)

        if model == "asr-v2":
            hf_results[model] = run_manifest_asr_benchmark(
                output_dir=output_dir,
                manifest_path=manifest_path,
                dataset_label=dataset_label,
                model_version="v2",
            )
        elif model == "asr-v3":
            hf_results[model] = run_manifest_asr_benchmark(
                output_dir=output_dir,
                manifest_path=manifest_path,
                dataset_label=dataset_label,
                model_version="v3",
            )
        elif model == "qwen3-f32":
            hf_results[model] = run_manifest_qwen3_benchmark(
                output_dir=output_dir,
                manifest_path=manifest_path,
                dataset_label=dataset_label,
                variant="f32",
                language=args.qwen3_language,
            )
        elif model == "qwen3-int8":
            hf_results[model] = run_manifest_qwen3_benchmark(
                output_dir=output_dir,
                manifest_path=manifest_path,
                dataset_label=dataset_label,
                variant="int8",
                language=args.qwen3_language,
            )
        else:
            # Guarded by parse_model_list.
            hf_results[model] = None

    combined_output = output_dir / "hf_benchmark_results.json"
    combined_output.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "mode": "hf-dataset",
                "dataset": metadata,
                "models": models,
                "results": hf_results,
            },
            indent=2,
        )
    )

    summarize_hf_results(hf_results)
    print("\n" + "=" * 60)
    print("HF benchmark complete!")
    print("=" * 60)
    print(f"Results saved to: {combined_output}")
    return 0


def run_builtin_benchmark_mode(args: argparse.Namespace, output_dir: Path, timestamp: str) -> int:
    # Determine which benchmarks to run
    run_all = not (args.asr_only or args.vad_only or args.diar_only)

    print("=" * 60)
    print("FluidAudio Benchmark Suite")
    print("=" * 60)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Output: {output_dir}")
    print(f"Time: {timestamp}")

    # Build first
    if not build_release():
        return 1

    results = {}

    # Run benchmarks
    if run_all or args.asr_only:
        results["asr"] = run_asr_benchmark(output_dir, args.quick)

    if run_all or args.vad_only:
        results["vad"] = run_vad_benchmark(output_dir, args.quick)

    if run_all or args.diar_only:
        results["diarization"] = run_diarization_benchmark(output_dir, args.quick)

    # Save combined results
    combined_output = output_dir / "benchmark_results.json"
    combined_output.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "mode": "quick" if args.quick else "full",
                "baselines": BASELINES,
                "results": results,
            },
            indent=2,
        )
    )

    # Compare against baselines
    compare_results(results)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print(f"Results saved to: {combined_output}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="FluidAudio Benchmark Suite")

    # Built-in benchmark mode arguments
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with smaller datasets")
    parser.add_argument("--asr-only", action="store_true", help="Run ASR benchmark only")
    parser.add_argument("--vad-only", action="store_true", help="Run VAD benchmark only")
    parser.add_argument("--diar-only", action="store_true", help="Run diarization benchmark only")

    # Shared arguments
    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    # HF custom dataset mode
    parser.add_argument("--hf-dataset", type=str, help="Hugging Face dataset id (enables HF mode)")
    parser.add_argument("--hf-config", type=str, default=None, help="Hugging Face dataset config name")
    parser.add_argument("--hf-split", type=str, default="train", help="HF split name (default: train)")
    parser.add_argument("--hf-audio-column", type=str, default=None, help="HF audio column (auto-detected by default)")
    parser.add_argument("--hf-text-column", type=str, default=None, help="HF transcript column (auto-detected)")
    parser.add_argument("--hf-max-samples", type=str, default="200", help="Rows to benchmark, or 'all'")
    parser.add_argument("--hf-shuffle", action="store_true", help="Shuffle before selecting rows")
    parser.add_argument("--hf-seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--models",
        type=str,
        default="asr-v3,asr-v2,qwen3-f32,qwen3-int8",
        help="Comma-separated models or 'all': asr-v2,asr-v3,qwen3-f32,qwen3-int8",
    )
    parser.add_argument(
        "--qwen3-language",
        type=str,
        default="vi",
        help="Qwen3 language hint for HF mode (e.g., vi, en, zh)",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Prepare manifest/audio only, skip benchmarks")
    parser.add_argument("--no-build", action="store_true", help="Skip swift release build step")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("benchmark-results") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.hf_dataset:
            exit_code = run_hf_benchmark_mode(args, output_dir, timestamp)
        else:
            exit_code = run_builtin_benchmark_mode(args, output_dir, timestamp)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

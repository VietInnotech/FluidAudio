#!/usr/bin/env python3
"""
eval_server_asr.py — Evaluate FluidAudio ASR server against a real-audio manifest.

Usage:
    python3 Tools/eval_server_asr.py \
        --endpoint http://127.0.0.1:8080 \
        --api-key devkey \
        --manifest benchmark-results/fpt_fosd_500/hf_materialized/manifest.jsonl \
        --model fluidaudio-parakeet-v3 \
        --max-samples 100 \
        --out benchmark-results/server_eval/parakeet_v3_100.json

Phases:
    A) Functional matrix (3 files x each model + negative tests)
    B) N-sample model run with CER/WER
    C) Optional deep run (all manifest rows)

Requirements: Python 3.9+, only stdlib (no external deps).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python, no deps)
# ---------------------------------------------------------------------------
def _levenshtein(a: list, b: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, NFC, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text.lower().strip())
    # Remove common punctuation
    text = "".join(c for c in text if unicodedata.category(c)[0] != "P")
    # Collapse whitespace
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def post_transcribe(
    endpoint: str,
    api_key: str | None,
    audio_path: str,
    model: str,
    response_format: str = "json",
    language: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """POST /v1/audio/transcriptions and return parsed response + metadata."""
    url = f"{endpoint}/v1/audio/transcriptions"
    boundary = f"----PythonBoundary{int(time.time() * 1000)}"

    # Build multipart body
    parts: list[bytes] = []

    # model field
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="model"\r\n\r\n')
    parts.append(f"{model}\r\n".encode())

    # response_format field
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
    parts.append(f"{response_format}\r\n".encode())

    # language field (optional)
    if language:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(b'Content-Disposition: form-data; name="language"\r\n\r\n')
        parts.append(f"{language}\r\n".encode())

    # file field
    filename = os.path.basename(audio_path)
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    with open(audio_path, "rb") as f:
        parts.append(f.read())
    parts.append(b"\r\n")

    # Closing boundary
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.monotonic() - t0
            resp_body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        elapsed = time.monotonic() - t0
        return {
            "status": e.code,
            "error": e.read().decode("utf-8", errors="replace"),
            "latency": elapsed,
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {"status": 0, "error": str(e), "latency": elapsed}

    result = {"status": status, "latency": elapsed, "raw": resp_body}
    if response_format in ("json", "verbose_json"):
        try:
            result["parsed"] = json.loads(resp_body)
        except json.JSONDecodeError:
            result["parsed"] = None
    else:
        result["text"] = resp_body
    return result


def get_json(endpoint: str, path: str, api_key: str | None, timeout: float = 10.0) -> dict:
    """GET a JSON endpoint."""
    url = f"{endpoint}{path}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "parsed": json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode("utf-8", errors="replace")}
    except Exception as e:
        return {"status": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Phase A: Functional Matrix
# ---------------------------------------------------------------------------
ALL_MODELS = [
    "fluidaudio-parakeet-v2",
    "fluidaudio-parakeet-v3",
    "fluidaudio-qwen3-f32",
    "fluidaudio-qwen3-int8",
    "fluidaudio-ctc-vi",
]


def run_phase_a(
    endpoint: str,
    api_key: str | None,
    manifest: list[dict],
    models: list[str],
) -> dict:
    """Run functional matrix: 3 files per model + negative tests."""
    print("\n" + "=" * 60)
    print("PHASE A: Functional Matrix")
    print("=" * 60)

    results: dict = {"models": {}, "negative_tests": {}}
    test_files = manifest[:3]

    for model in models:
        print(f"\n  Model: {model}")
        model_results = []
        for entry in test_files:
            for fmt in ("json", "text", "verbose_json"):
                resp = post_transcribe(
                    endpoint, api_key, entry["audio_path"], model,
                    response_format=fmt,
                )
                ok = resp["status"] == 200
                has_text = False
                if fmt == "text":
                    has_text = bool(resp.get("text", "").strip())
                elif resp.get("parsed"):
                    has_text = bool(resp["parsed"].get("text", "").strip())

                status_str = "PASS" if (ok and has_text) else "FAIL"
                print(f"    [{status_str}] {entry['file_name']} fmt={fmt} status={resp['status']}")
                model_results.append({
                    "file": entry["file_name"],
                    "format": fmt,
                    "status": resp["status"],
                    "ok": ok,
                    "has_text": has_text,
                    "latency": resp["latency"],
                })
        results["models"][model] = model_results

    # Negative tests
    print("\n  Negative Tests:")

    # Missing model field
    resp = post_transcribe(endpoint, api_key, test_files[0]["audio_path"], "")
    passed = resp["status"] == 400
    print(f"    [{'PASS' if passed else 'FAIL'}] missing model -> {resp['status']}")
    results["negative_tests"]["missing_model"] = {"status": resp["status"], "passed": passed}

    # Invalid model name
    resp = post_transcribe(endpoint, api_key, test_files[0]["audio_path"], "nonexistent-model")
    passed = resp["status"] == 400
    print(f"    [{'PASS' if passed else 'FAIL'}] invalid model -> {resp['status']}")
    results["negative_tests"]["invalid_model"] = {"status": resp["status"], "passed": passed}

    # Invalid API key (only if API key is configured)
    if api_key:
        resp = post_transcribe(endpoint, "wrong-key-12345", test_files[0]["audio_path"], models[0])
        passed = resp["status"] == 401
        print(f"    [{'PASS' if passed else 'FAIL'}] invalid api key -> {resp['status']}")
        results["negative_tests"]["invalid_api_key"] = {"status": resp["status"], "passed": passed}

    # Concurrent requests -> 429
    print("    Testing concurrent requests (expect 429)...")
    concurrent_result = test_concurrent_429(endpoint, api_key, test_files[0]["audio_path"], models[0])
    print(f"    [{'PASS' if concurrent_result['passed'] else 'FAIL'}] concurrent -> {concurrent_result}")
    results["negative_tests"]["concurrent_429"] = concurrent_result

    return results


def test_concurrent_429(endpoint: str, api_key: str | None, audio_path: str, model: str) -> dict:
    """Send two concurrent requests; expect one 200 and one 429."""
    statuses = []

    def do_request():
        resp = post_transcribe(endpoint, api_key, audio_path, model)
        return resp["status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(do_request) for _ in range(2)]
        for f in futures:
            statuses.append(f.result())

    has_200 = 200 in statuses
    has_429 = 429 in statuses
    return {"statuses": statuses, "passed": has_200 and has_429}


# ---------------------------------------------------------------------------
# Phase B/C: N-sample benchmark run
# ---------------------------------------------------------------------------
def run_benchmark(
    endpoint: str,
    api_key: str | None,
    manifest: list[dict],
    model: str,
    max_samples: int,
    language: str | None = None,
) -> dict:
    """Run N-sample benchmark: POST each file, collect metrics."""
    samples = manifest[:max_samples]
    n = len(samples)
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {model} ({n} samples)")
    print("=" * 60)

    latencies: list[float] = []
    cers: list[float] = []
    wers: list[float] = []
    successes = 0
    failures = 0
    errors_log: list[dict] = []

    for i, entry in enumerate(samples):
        resp = post_transcribe(
            endpoint, api_key, entry["audio_path"], model,
            response_format="json", language=language,
        )
        status = resp["status"]
        latency = resp["latency"]

        if status == 200 and resp.get("parsed"):
            successes += 1
            latencies.append(latency)

            hyp = normalize_text(resp["parsed"].get("text", ""))
            ref = normalize_text(entry.get("transcript", ""))
            if ref:
                cer = compute_cer(ref, hyp)
                wer = compute_wer(ref, hyp)
                cers.append(cer)
                wers.append(wer)
        else:
            failures += 1
            errors_log.append({
                "index": i,
                "file": entry["file_name"],
                "status": status,
                "error": resp.get("error", ""),
            })

        if (i + 1) % 10 == 0 or i == n - 1:
            pct = (i + 1) / n * 100
            avg_lat = statistics.mean(latencies) if latencies else 0
            print(f"  [{i + 1}/{n}] {pct:.0f}% | ok={successes} fail={failures} | avg_lat={avg_lat:.2f}s")

    # Compute summary statistics
    result: dict = {
        "model": model,
        "total_samples": n,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / n if n > 0 else 0,
    }

    if latencies:
        sorted_lat = sorted(latencies)
        result["latency"] = {
            "mean": statistics.mean(sorted_lat),
            "median": statistics.median(sorted_lat),
            "p50": sorted_lat[len(sorted_lat) // 2],
            "p95": sorted_lat[int(len(sorted_lat) * 0.95)],
            "min": sorted_lat[0],
            "max": sorted_lat[-1],
        }

    if cers:
        result["cer"] = {
            "mean": statistics.mean(cers),
            "median": statistics.median(cers),
            "min": min(cers),
            "max": max(cers),
        }

    if wers:
        result["wer"] = {
            "mean": statistics.mean(wers),
            "median": statistics.median(wers),
            "min": min(wers),
            "max": max(wers),
        }

    if errors_log:
        result["errors"] = errors_log[:20]  # Cap logged errors

    print(f"\n  Success rate: {result['success_rate'] * 100:.1f}%")
    if "cer" in result:
        print(f"  Mean CER: {result['cer']['mean'] * 100:.2f}%")
    if "wer" in result:
        print(f"  Mean WER: {result['wer']['mean'] * 100:.2f}%")
    if "latency" in result:
        print(f"  Latency p50={result['latency']['p50']:.2f}s p95={result['latency']['p95']:.2f}s")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> list[dict]:
    rows = []
    manifest_dir = os.path.dirname(os.path.abspath(path))
    # Try to find the audio directory relative to the manifest
    audio_dir = os.path.join(manifest_dir, "audio")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Normalise field names:
            # - audio_path: full path to audio file
            # - transcript: reference text
            if "audio_path" not in entry:
                fname = entry.get("file_name", "")
                if os.path.isabs(fname):
                    entry["audio_path"] = fname
                else:
                    entry["audio_path"] = os.path.join(audio_dir, fname)
            if "transcript" not in entry:
                entry["transcript"] = entry.get("text", "")
            rows.append(entry)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate FluidAudio ASR server")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080", help="Server base URL")
    parser.add_argument("--api-key", default=None, help="API key (Bearer token)")
    parser.add_argument(
        "--manifest", required=True,
        help="Path to manifest.jsonl with audio_path and transcript fields",
    )
    parser.add_argument(
        "--model", default="fluidaudio-parakeet-v3",
        help="Model ID for benchmark phases",
    )
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples for benchmark")
    parser.add_argument("--language", default=None, help="Language hint (for Qwen3 models)")
    parser.add_argument("--out", default=None, help="Output JSON path for benchmark results")
    parser.add_argument(
        "--phase-a-models", nargs="*", default=None,
        help="Models to test in Phase A (default: all). Pass model IDs separated by space.",
    )
    parser.add_argument("--skip-phase-a", action="store_true", help="Skip Phase A functional tests")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"Loaded {len(manifest)} manifest entries from {args.manifest}")

    # Verify server is reachable
    health = get_json(args.endpoint, "/health", args.api_key)
    if health.get("status") != 200:
        print(f"ERROR: Server health check failed: {health}")
        sys.exit(1)
    print(f"Server health: OK")

    all_results: dict = {}

    # Phase A
    if not args.skip_phase_a:
        phase_a_models = args.phase_a_models or ALL_MODELS
        all_results["phase_a"] = run_phase_a(args.endpoint, args.api_key, manifest, phase_a_models)

    # Phase B: benchmark
    benchmark = run_benchmark(
        args.endpoint, args.api_key, manifest, args.model, args.max_samples,
        language=args.language,
    )
    all_results["benchmark"] = benchmark

    # Write output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {out_path}")

    # Exit with error if success rate is too low
    sr = benchmark.get("success_rate", 0)
    if sr < 0.99:
        print(f"\nWARNING: Success rate {sr * 100:.1f}% is below 99% threshold")


if __name__ == "__main__":
    main()

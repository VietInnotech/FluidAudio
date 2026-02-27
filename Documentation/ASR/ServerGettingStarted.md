# ASR Server — Getting Started

FluidAudio includes a self-hosted ASR server with an OpenAI-compatible `/v1/audio/transcriptions` endpoint. It is a single-client server designed to run locally on a Mac with Apple Silicon.

## Quick Start

### 1. Build

```bash
swift build -c release --product fluidaudio-server
```

### 2. Run

```bash
# Without authentication (open access):
.build/release/fluidaudio-server

# With an API key:
FLUIDAUDIO_SERVER_API_KEY=devkey .build/release/fluidaudio-server
```

### 3. Transcribe

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "file=@audio.wav"
```

Response:

```json
{"text": "the quick brown fox jumps over the lazy dog"}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLUIDAUDIO_SERVER_HOST` | `127.0.0.1` | Bind address |
| `FLUIDAUDIO_SERVER_PORT` | `8080` | Listen port |
| `FLUIDAUDIO_SERVER_API_KEY` | _(none)_ | Optional Bearer token. If unset, no auth is required. |
| `FLUIDAUDIO_SERVER_MAX_UPLOAD_MB` | `128` | Maximum upload size in MB |
| `FLUIDAUDIO_SERVER_MAX_AUDIO_SECONDS` | `900` | Maximum audio duration in seconds (15 min) |

## Endpoints

### `GET /health`

Always returns `200` with `{"status":"ok"}`. Not subject to authentication.

### `GET /v1/models`

Returns a list of available models in OpenAI-compatible format.

### `POST /v1/audio/transcriptions`

Multipart form fields:

| Field | Required | Description |
|---|---|---|
| `file` | Yes | Audio file (WAV, MP3, M4A, FLAC, CAF, etc.) |
| `model` | Yes | Model ID (see below) |
| `language` | No | Language hint (ISO code, e.g. `vi`, `en`). Only used by Qwen3 models. |
| `response_format` | No | `json` (default), `text`, or `verbose_json` |

## Available Models

| Model ID | Engine | Notes |
|---|---|---|
| `fluidaudio-parakeet-v2` | Parakeet TDT v2 (0.6B) | English-optimized |
| `fluidaudio-parakeet-v3` | Parakeet TDT v3 (0.6B) | 25 European languages |
| `fluidaudio-qwen3-f32` | Qwen3-ASR (FP16) | Multilingual, requires macOS 15+ |
| `fluidaudio-qwen3-int8` | Qwen3-ASR (Int8) | Lower memory, requires macOS 15+ |
| `fluidaudio-ctc-vi` | Parakeet CTC Vietnamese | Vietnamese + English code-switching |

Models are downloaded automatically from HuggingFace on first use.

## Concurrency

The server processes **one transcription at a time**. If a second request arrives while one is in progress, it returns `429 Too Many Requests`. This avoids memory pressure from loading multiple CoreML models simultaneously.

## Model Lifecycle

Only one model backend is kept in memory. When you switch models (e.g., from `fluidaudio-parakeet-v3` to `fluidaudio-qwen3-f32`), the old model is unloaded first. The first request for a new model will include the model loading time.

## Response Formats

**`json`** (default):
```json
{"text": "transcribed text here"}
```

**`text`**:
```
transcribed text here
```

**`verbose_json`**:
```json
{
  "text": "transcribed text here",
  "model": "fluidaudio-parakeet-v3",
  "duration": 5.2,
  "processing_time": 0.31
}
```

## Error Responses

Errors follow a standard JSON format:

```json
{
  "error": {
    "message": "Missing required field: file",
    "type": "invalid_request_error",
    "code": null
  }
}
```

| Status | Meaning |
|---|---|
| `400` | Invalid parameters or unknown model |
| `401` | Missing or invalid API key |
| `413` | Upload or audio duration exceeds limits |
| `415` | Request is not multipart/form-data |
| `429` | Server is busy processing another request |
| `500` | Internal error or model failure |

## Evaluation Script

The included `Tools/eval_server_asr.py` script runs automated accuracy and performance benchmarks against a manifest of real audio files:

```bash
python3 Tools/eval_server_asr.py \
    --endpoint http://127.0.0.1:8080 \
    --api-key devkey \
    --manifest benchmark-results/fpt_fosd_500/hf_materialized/manifest.jsonl \
    --model fluidaudio-parakeet-v3 \
    --max-samples 100 \
    --out benchmark-results/server_eval/parakeet_v3_100.json
```

The script runs:
- **Phase A**: Functional matrix (3 files × all models + negative tests)
- **Phase B**: N-sample benchmark with CER/WER metrics

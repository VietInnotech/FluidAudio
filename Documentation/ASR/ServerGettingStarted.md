# ASR Server — Getting Started

FluidAudio includes a self-hosted ASR server with an OpenAI-compatible `/v1/audio/transcriptions` endpoint. It is designed for local deployment on a Mac with Apple Silicon, processing one transcription at a time.

## Features

- **OpenAI-compatible API** — drop-in replacement for OpenAI's Whisper API
- **Multiple Models** — Parakeet (English & 25 European languages), Qwen3 (multilingual), CTC (Vietnamese)
- **Fast** — RTFx 200–250x on M4 Pro (real-time capable)
- **Automatic Model Download** — Models fetched from HuggingFace on first use
- **Thread-Safe** — Actor-based design with proper concurrency control
- **Verbose Logging** — detailed per-request and model-loading diagnostics

## Quick Start

### 1. Build the Server

```bash
cd /path/to/FluidAudio
swift build -c release --product fluidaudio-server
```

Release builds are significantly faster than debug builds (especially for transcription). Use `--product fluidaudio-server` (not `--target`) — it ensures the binary is linked at `.build/release/fluidaudio-server`.

> **Note:** After `swift package clean`, the `.build/release/` symlink is removed. Running the build command above recreates it.

### 2. Configure (Optional)

Create a `.env` file in your working directory:

```bash
cat > .env <<'EOF'
FLUIDAUDIO_SERVER_HOST=127.0.0.1
FLUIDAUDIO_SERVER_PORT=8080
FLUIDAUDIO_SERVER_API_KEY=devkey
FLUIDAUDIO_SERVER_MAX_UPLOAD_MB=128
FLUIDAUDIO_SERVER_MAX_AUDIO_SECONDS=1800
FLUIDAUDIO_LOG_LEVEL=info
EOF
```

The server automatically loads this file on startup.

### 3. Run the Server

```bash
cd /path/to/FluidAudio
.build/release/fluidaudio-server
```

If `.build/release/fluidaudio-server` is not found (e.g. after `swift package clean`), rebuild with:
```bash
swift build -c release --product fluidaudio-server
```

Expected output:
```
2026-03-02T14:48:53+0700 info FluidAudioServer: Starting FluidAudio ASR Server
2026-03-02T14:48:53+0700 info FluidAudioServer: Host: 127.0.0.1
2026-03-02T14:48:53+0700 info FluidAudioServer: Port: 8080
2026-03-02T14:48:53+0700 info FluidAudioServer: Max upload: 128 MB
2026-03-02T14:48:53+0700 info FluidAudioServer: Max audio duration: 1800s
2026-03-02T14:48:53+0700 info FluidAudioServer: API key: configured
2026-03-02T14:48:53+0700 info FluidAudioServer: Server ready at http://127.0.0.1:8080
```

### Running in the Background

```bash
.build/release/fluidaudio-server > server.log 2>&1 &
echo $! > server.pid
```

This stores the process ID in `server.pid` for later shutdown.

### Explore the API

Once the server is running, open the interactive Swagger UI in your browser:

```
http://localhost:8080/swagger
```

Or fetch the raw OpenAPI 3.1.0 spec:

```bash
curl http://localhost:8080/openapi.json | python3 -m json.tool
```

## Server Management

### Check if Server is Running

```bash
ps aux | grep fluidaudio-server | grep -v grep
```

Or check the port directly:

```bash
lsof -i :8080
```

### Stop the Server

If running in foreground, press **Ctrl+C**.

If running in background:

```bash
# If you saved the PID:
kill $(cat server.pid)

# Or find and kill:
pkill -f "fluidaudio-server"

# Force kill (if graceful shutdown hangs):
pkill -9 -f "fluidaudio-server"
```

### Restart the Server

```bash
pkill -f "fluidaudio-server"
sleep 1
.build/release/fluidaudio-server
```

### 4. Test the Server

In another terminal:

```bash
# Health check (no auth required)
curl http://127.0.0.1:8080/health
```

Response:
```json
{"status":"ok"}
```

Now transcribe audio:

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "file=@audio.wav"
```

Response:
```json
{"text": "the quick brown fox jumps over the lazy dog"}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLUIDAUDIO_SERVER_HOST` | `127.0.0.1` | Bind address |
| `FLUIDAUDIO_SERVER_PORT` | `8080` | Listen port |
| `FLUIDAUDIO_SERVER_API_KEY` | _(none)_ | Optional Bearer token. If unset, no auth required. |
| `FLUIDAUDIO_SERVER_MAX_UPLOAD_MB` | `128` | Maximum upload size in MB |
| `FLUIDAUDIO_SERVER_MAX_AUDIO_SECONDS` | `1800` | Maximum audio duration in seconds (30 min) |
| `FLUIDAUDIO_LOG_LEVEL` | `info` | Log verbosity: `debug`, `info`, `warning`, `error`, `critical` |

### Authentication

**Public Endpoints** (no authentication required):
- `GET /health` — Server status check
- `GET /swagger` — Interactive Swagger UI documentation
- `GET /openapi.json` — OpenAPI 3.1.0 specification

**Protected Endpoints** (Bearer token required if `FLUIDAUDIO_SERVER_API_KEY` is set):
- `GET /v1/models` — List available models
- `POST /v1/audio/transcriptions` — Transcribe audio

**If `FLUIDAUDIO_SERVER_API_KEY` is not set:**
- All endpoints are accessible without authentication
- Public endpoints still work normally
- Protected endpoints work without Bearer token

**If `FLUIDAUDIO_SERVER_API_KEY` is set:**
- Public endpoints remain unauthenticated and always accessible
- Protected endpoints require Bearer token in request header
- Example: `curl -H "Authorization: Bearer devkey" http://localhost:8080/v1/models`

### Loading Configuration

The server loads configuration in this order (later values override earlier):

1. Default hardcoded values
2. `.env` file in the working directory (if present)
3. Environment variables (shell or command-line)

Example `.env` file for common scenarios:

**Development** (verbose logging, high limits):
```bash
FLUIDAUDIO_SERVER_HOST=127.0.0.1
FLUIDAUDIO_SERVER_PORT=8080
FLUIDAUDIO_SERVER_MAX_UPLOAD_MB=256
FLUIDAUDIO_SERVER_MAX_AUDIO_SECONDS=3600
FLUIDAUDIO_LOG_LEVEL=debug
```

**Production** (tight security):
```bash
FLUIDAUDIO_SERVER_HOST=0.0.0.0
FLUIDAUDIO_SERVER_PORT=8080
FLUIDAUDIO_SERVER_API_KEY=your-secret-key-here
FLUIDAUDIO_SERVER_MAX_UPLOAD_MB=64
FLUIDAUDIO_SERVER_MAX_AUDIO_SECONDS=300
FLUIDAUDIO_LOG_LEVEL=warning
```

## Quick Testing Guide

### Using Swagger UI (Recommended for Exploration)

1. Start the server
2. Open your browser to `http://localhost:8080/swagger`
3. Click on `/v1/audio/transcriptions` endpoint
4. Click **"Try it out"**
5. Enter your Bearer token in the **"Authorization"** field (e.g., `Bearer devkey`)
6. Select a model (e.g., `fluidaudio-parakeet-v3`)
7. Upload an audio file
8. Click **"Execute"**

The response will show transcription text, processing metrics (duration, RTFx), and error details if any.

### Using cURL (Command Line)

To test the server with sample audio files:

```bash
# Start server
cd /Users/vit/offasr/FluidAudio
.build/release/fluidaudio-server &
sleep 2

# Test CTC-VI (Vietnamese, small file)
echo "Testing CTC-VI..."
curl -s -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-ctc-vi" \
  -F "response_format=verbose_json" \
  -F "file=@Tests/weanxinviec.mp3" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'RTFx: {d[\"duration\"]/d[\"processing_time\"]:.1f}x')"

# Test Parakeet v3 (larger file, multilingual audio)
echo "Testing Parakeet v3..."
curl -s -X POST http://localhost:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "response_format=verbose_json" \
  -F "file=@Tests/bomman.mp3" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'RTFx: {d[\"duration\"]/d[\"processing_time\"]:.1f}x')"

# Stop server
pkill -f "fluidaudio-server"
```

**Expected Results:**
- ✓ CTC-VI: completes in ~2-3 seconds with Vietnamese text
- ✓ Parakeet v3: completes in ~7-10 seconds with transcribed multilingual audio
- ✓ Qwen3 F32: completes in ~9 seconds (3s load + 6s inference for 30s audio)
- ✓ Qwen3 Int8: completes in ~29 seconds (2s load + 27s inference for 30s audio)

| Use Case | Recommended Model | RTFx | Status |
|---|---|---|---|
| English or 25 European languages | `fluidaudio-parakeet-v3` | 200x | ✓ Production ready |
| Vietnamese (pure or code-switching) | `fluidaudio-ctc-vi` | 80x | ✓ Tested & working |
| Multilingual (Chinese, Japanese, etc.) | `fluidaudio-parakeet-v3` | 200x | ✓ Works for many languages |
| Multilingual (30 languages, LLM-based) | `fluidaudio-qwen3-f32` | 5x | ✓ Working, higher quality |

### Parakeet Models (multilingual, low-latency) ✓ Tested & Working

| Model ID | Language | Size | Speed | Notes |
|---|---|---|---|---|
| `fluidaudio-parakeet-v2` | English only | 0.6B params | ~250x RTFx | Baseline Parakeet |
| `fluidaudio-parakeet-v3` | 25 European langs | 0.6B params | ~200x RTFx | ✓ **Recommended for production** |

**Supported languages in Parakeet v3:** English, German, French, Italian, Spanish, Dutch, Portuguese, Polish, Russian, Turkish, Swedish, Norwegian, Danish, Finnish, Czech, Slovak, Hungarian, Romanian, Greek, Bulgarian, Croatian, Slovenian, Estonian, Latvian, Lithuanian.

### CTC Models (specialized) ✓ Tested & Working

| Model ID | Language | Specialization | Speed | Notes |
|---|---|---|---|---|
| `fluidaudio-ctc-vi` | Vietnamese + English | Code-switching | ~80x RTFx | ✓ **Works perfectly** |

**Best for:** Vietnamese audio with code-switching to English. Also works for pure Vietnamese or pure English.

### Qwen3 Models (multilingual, LLM-based) ✓ Working

| Model ID | Precision | Size | Speed | Notes |
|---|---|---|---|---|
| `fluidaudio-qwen3-f32` | Float32 (FP16) | 0.6B params | ~5x RTFx | ✓ Best quality, 3s model load |
| `fluidaudio-qwen3-int8` | Int8 quantized | 0.6B params | ~1x RTFx | ✓ Lower RAM (~2 GB), slower inference |

Qwen3 models support 30 languages including Chinese, Japanese, Korean, Arabic, Vietnamese, and more. They use an autoregressive LLM decoder, so they are much slower than Parakeet but may produce higher quality output for languages Parakeet doesn't cover.

**Note:** First load takes 2-3 seconds while CoreML compiles and caches the models. Memory usage peaks at ~3 GB during inference.

### Model Notes

- **First Use** — models are downloaded automatically from HuggingFace (~50–200 MB depending on model). Subsequent requests use the cached version.
- **Single Active Model** — only one model backend is kept in memory. Switching models unloads the previous one.
- **Language Hints** — Qwen3 models accept a `language` parameter to improve accuracy for known languages.

## Making Requests

### Basic Request

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "file=@audio.mp3"
```

### With Language Hint (Qwen3 models only)

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-qwen3-f32" \
  -F "language=vi" \
  -F "file=@vietnamese-audio.wav"
```

### With Custom Response Format

```bash
# Verbose JSON with metrics
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "response_format=verbose_json" \
  -F "file=@audio.wav"
```

## Response Formats

### JSON (default)

```json
{"text": "the quick brown fox"}
```

### Text

```
the quick brown fox
```

### Verbose JSON

```json
{
  "text": "the quick brown fox",
  "model": "fluidaudio-parakeet-v3",
  "duration": 2.5,
  "processing_time": 0.012
}
```

## API Endpoints

### `GET /health`

Health check endpoint (no authentication required).

**Response:**
```json
{"status":"ok"}
```

**Example:**
```bash
curl http://127.0.0.1:8080/health
```

### `GET /swagger`

Interactive Swagger UI documentation (no authentication required).

Serves a full OpenAPI-compliant Swagger UI interface for exploring and testing the API directly in your browser.

**Example:**
```bash
# Open in your browser
open http://127.0.0.1:8080/swagger

# Or fetch the HTML
curl http://127.0.0.1:8080/swagger
```

### `GET /openapi.json`

OpenAPI 3.1.0 specification in JSON format (no authentication required).

Returns the complete OpenAPI specification describing all endpoints, request/response schemas, authentication methods, and error codes.

**Example:**
```bash
curl http://127.0.0.1:8080/openapi.json | python3 -m json.tool
```

**Use Cases:**
- Load into Postman, Insomnia, or other API clients
- Generate client libraries with `openapi-generator`
- Document API integrations programmatically
- IDE integration (VS Code, etc.) for autocomplete and validation

### `GET /v1/models`

List available models in OpenAI-compatible format.

**Authentication:** Requires Bearer token if `FLUIDAUDIO_SERVER_API_KEY` is set.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "fluidaudio-parakeet-v2",
      "owned_by": "fluidaudio",
      "object": "model",
      "created": 1772438622
    },
    ...
  ]
}
```

**Example:**
```bash
curl -H "Authorization: Bearer devkey" \
  http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

### `POST /v1/audio/transcriptions`

Transcribe audio using the specified model.

**Authentication:** Requires Bearer token if `FLUIDAUDIO_SERVER_API_KEY` is set.

**Request:**

Multipart form fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | Audio file (WAV, MP3, M4A, FLAC, CAF, etc.) |
| `model` | string | Yes | Model ID (e.g. `fluidaudio-parakeet-v3`) |
| `language` | string | No | Language hint (ISO 639-1 code: `en`, `vi`, `fr`, etc.). Only used by Qwen3 models. |
| `response_format` | string | No | `json` (default), `text`, or `verbose_json` |

**Supported Audio Formats:**
- WAV (PCM, various bit depths)
- MP3
- M4A (AAC)
- FLAC
- CAF
- OGG
- MP4

Audio is automatically resampled to 16 kHz mono if needed.

**Examples:**

```bash
# Basic transcription
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "file=@audio.wav"

# With language hint (Qwen3 only)
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-qwen3-f32" \
  -F "language=vi" \
  -F "file=@audio.wav"

# Verbose response with metrics
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "response_format=verbose_json" \
  -F "file=@audio.wav"
```

**Responses:**

Success (JSON):
```json
{"text": "the quick brown fox"}
```

Success (verbose JSON):
```json
{
  "text": "the quick brown fox",
  "model": "fluidaudio-parakeet-v3",
  "duration": 2.5,
  "processing_time": 0.012
}
```

## Error Responses

All errors are returned as JSON with HTTP status codes:

```json
{
  "error": {
    "message": "descriptive error message",
    "type": "error_category",
    "code": "error_code_or_null"
  }
}
```

### Error Status Codes

| Status | Type | Cause | Solution |
|---|---|---|---|
| `400` | `invalid_request_error` | Missing field, unknown model, or invalid parameters | Check request format and model ID (use GET /v1/models) |
| `401` | `authentication_error` | Missing or invalid API key | Add `-H "Authorization: Bearer <key>"` |
| `413` | `invalid_request_error` | File too large or audio too long | Reduce file size or duration (set smaller `MAX_UPLOAD_MB` or `MAX_AUDIO_SECONDS`) |
| `415` | `invalid_request_error` | Request is not multipart/form-data | Use `-F` with curl, not `-d` |
| `429` | `rate_limit_error` | Server processing another request | Single-client server; wait for previous request to complete |
| `500` | `server_error` | Internal error or model failure | Check server logs and restart if needed |

### Example Error

```bash
$ curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "model=invalid-model" \
  -F "file=@audio.wav"

# Response (400):
{
  "error": {
    "message": "Model 'invalid-model' not found. Use GET /v1/models to list available models.",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

## Server Logging

The server logs all requests, responses, and model operations with configurable verbosity.

### Log Levels

Set `FLUIDAUDIO_LOG_LEVEL` to control verbosity (default: `info`):

- `debug` — detailed diagnostics (model operations, request processing)
- `info` — key events (request start/end, model loading)
- `warning` — errors and rejections (busy server, auth failures)
- `error` — failures only
- `critical` — critical failures only

Example:
```bash
FLUIDAUDIO_LOG_LEVEL=debug .build/release/fluidaudio-server
```

### Sample Logs

**Info level** (default):
```
2026-03-02T14:48:53+0700 info FluidAudioServer: Starting FluidAudio ASR Server
2026-03-02T14:48:53+0700 info FluidAudioServer: Server ready at http://127.0.0.1:8080
2026-03-02T14:48:55+0700 info Hummingbird: → POST /v1/audio/transcriptions [2.1 MB]
2026-03-02T14:48:55+0700 info FluidAudioServer.TranscriptionService: transcribing: model=fluidaudio-parakeet-v3 lang=auto file=2.1 MB audio=47.3s
2026-03-02T14:48:56+0700 info FluidAudioServer.TranscriptionService: model ready: fluidaudio-parakeet-v3 (2.3s)
2026-03-02T14:49:01+0700 info FluidAudioServer.TranscriptionService: done: 47.3s audio → 0.226s (RTFx 209.3x) 412 chars
2026-03-02T14:49:01+0700 info Hummingbird: ← 200 OK 6.123s
```

**Debug level** (includes step-by-step model loading):
```
2026-03-02T14:48:55+0700 info RequestLoggingMiddleware: → POST /v1/audio/transcriptions [2.1 MB]
2026-03-02T14:48:55+0700 info TranscriptionService: switching model: none → fluidaudio-parakeet-v3
2026-03-02T14:48:55+0700 info TranscriptionService: downloading parakeet-tdt-v3 models…
2026-03-02T14:48:56+0700 info TranscriptionService: initializing parakeet-tdt-v3…
2026-03-02T14:48:56+0700 info TranscriptionService: loading VAD…
2026-03-02T14:48:56+0700 info TranscriptionService: model ready: fluidaudio-parakeet-v3 (2.3s)
```

### Interpreting Perf Metrics

From the `done:` log line:
- `47.3s audio` — duration of the audio file
- `→ 0.226s` — how long transcription took
- `RTFx 209.3x` — real-time factor (audio_duration / processing_time)
  - RTFx > 1x = real-time or faster
  - RTFx 209x = 209 seconds of audio per 1 second of processing

## Concurrency Model

The server processes **exactly one transcription at a time**. This design:

- ✅ Keeps memory usage predictable (only one CoreML model loaded)
- ✅ Avoids ANE (Apple Neural Engine) contention
- ✅ Simplifies request lifecycle management

When a second request arrives during processing:

```json
{
  "error": {
    "message": "Server is currently processing another request. Try again later.",
    "type": "rate_limit_error"
  }
}
```

**Response:** HTTP 429 Too Many Requests

**Solution:** Retry with exponential backoff:
```bash
# Simple retry loop
for i in {1..5}; do
  curl ... && break
  sleep $((2 ** i))
done
```

## Model Management

### Automatic Download

Models are downloaded automatically from HuggingFace on first use. Download times:

| Model | Size | Typical Time |
|---|---|---|
| Parakeet v2 | 50 MB | 10–30s (depending on network) |
| Parakeet v3 | 60 MB | 15–40s |
| Qwen3 F32 | ~1.5 GB | 30–120s |
| Qwen3 Int8 | ~1.2 GB | 30–90s |

Subsequent requests use the cached version (no download).

### Model Cache Location

Models are cached at:
```
~/Library/Caches/FluidAudio/
```

To clear the cache (will re-download on next use):
```bash
rm -rf ~/Library/Caches/FluidAudio/
```

### Model Switching

Only one model backend is active in memory. When switching models:

1. Previous model is unloaded
2. New model is downloaded (if needed)
3. New model is loaded into memory

This means the **first request to a new model includes model loading time**. Subsequent requests reuse the loaded model.

## Troubleshooting

### The request hangs forever

**Symptom:** No logs after `→ POST /v1/audio/transcriptions` line

**Possible causes:**

1. **Models downloading** — First-time model use triggers large download (~50–200 MB)
   - Solution: Monitor network activity; wait for completion
   - Enable debug logging: `FLUIDAUDIO_LOG_LEVEL=debug`

2. **Server is busy** — Another transcription is in progress
   - Solution: Wait for completion or check server logs

3. **Audio conversion stalled** — MP3/M4A decoding can be slow on first use
   - Solution: Try a simpler WAV file; use debug logging

**Debug:**
```bash
FLUIDAUDIO_LOG_LEVEL=debug .build/release/fluidaudio-server
# In another terminal:
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer devkey" \
  -F "model=fluidaudio-parakeet-v3" \
  -F "file=@audio.wav"
```

Watch the server logs to see where it's stuck.

### Server crashes on startup

**Symptom:** `error: ...` message, then exit

**Common causes:**

1. **Invalid `FLUIDAUDIO_LOG_LEVEL`** — must be one of: `debug`, `info`, `warning`, `error`, `critical`
   - Solution: Fix the env var and restart

2. **Port already in use** — another process is on port 8080
   - Solution: Check with `lsof -i :8080` or use a different port: `FLUIDAUDIO_SERVER_PORT=9000`

3. **File permissions** — cannot write to cache directory
   - Solution: Check permissions on `~/Library/Caches/FluidAudio/`

### Model not found error

**Symptom:** HTTP 400 with `"code": "model_not_found"`

**Solution:** Check available models:
```bash
curl -H "Authorization: Bearer devkey" http://127.0.0.1:8080/v1/models
```

Use the exact `id` from that list.

### API key not working

**Symptom:** HTTP 401 with `"message": "Invalid API key"`

**Solution:** Ensure Bearer token is correct:
```bash
# Correct:
curl -H "Authorization: Bearer devkey" ...

# Incorrect:
curl -H "Authorization: devkey" ...  # Missing "Bearer"
curl -H "X-API-Key: devkey" ...      # Wrong header name
```

### 429 Too Many Requests

**Symptom:** Server is busy

**Cause:** Single-client server; another transcription is in progress

**Solution:** Implement retry logic with exponential backoff:
```python
import requests
import time

for attempt in range(5):
    response = requests.post(
        "http://127.0.0.1:8080/v1/audio/transcriptions",
        headers={"Authorization": "Bearer devkey"},
        files={"file": open("audio.wav", "rb"), "model": "fluidaudio-parakeet-v3"}
    )
    if response.status_code != 429:
        break
    wait_time = 2 ** attempt
    print(f"Server busy, retrying in {wait_time}s...")
    time.sleep(wait_time)
```

### Qwen3 Models Are Slow

**Symptom:** Qwen3 requests take much longer than Parakeet (~6s for 30s audio vs ~0.2s)

**This is expected.** Qwen3 uses autoregressive LLM decoding (generating tokens one at a time), while Parakeet uses a parallel CTC/TDT architecture. Qwen3 F32 achieves ~5x RTFx; Int8 achieves ~1x RTFx.

Memory usage peaks at ~3 GB during Qwen3 inference — this is normal for a 0.6B parameter model.

**When to use Qwen3:**
- Languages not supported by Parakeet (Chinese, Japanese, Korean, Arabic, etc.)
- Higher transcription quality needed for specific languages

**When to use Parakeet instead:**
- Real-time / low-latency requirements (200x RTFx)
- English or European languages
- Production workloads with high throughput

## Evaluation & Benchmarking

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
- **Phase A**: Functional matrix (all models + error cases)
- **Phase B**: N-sample benchmark with CER/WER metrics

## Client Examples

### Python

```python
import requests
import json

MODEL = "fluidaudio-parakeet-v3"
ENDPOINT = "http://127.0.0.1:8080/v1/audio/transcriptions"
API_KEY = "devkey"

with open("audio.wav", "rb") as f:
    response = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={
            "file": f,
            "model": MODEL,
            "response_format": "verbose_json"
        }
    )

data = response.json()
print(f"Text: {data['text']}")
print(f"RTFx: {data['duration'] / data['processing_time']:.1f}x")
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const model = "fluidaudio-parakeet-v3";
const endpoint = "http://127.0.0.1:8080/v1/audio/transcriptions";
const apiKey = "devkey";

const form = new FormData();
form.append('file', fs.createReadStream('audio.wav'));
form.append('model', model);
form.append('response_format', 'verbose_json');

axios.post(endpoint, form, {
    headers: {
        ...form.getHeaders(),
        'Authorization': `Bearer ${apiKey}`
    }
}).then(response => {
    console.log(`Text: ${response.data.text}`);
    console.log(`RTFx: ${(response.data.duration / response.data.processing_time).toFixed(1)}x`);
}).catch(error => console.error(error));
```

### Shell/cURL

```bash
#!/bin/bash

MODEL="fluidaudio-parakeet-v3"
ENDPOINT="http://127.0.0.1:8080/v1/audio/transcriptions"
API_KEY="devkey"
AUDIO_FILE="audio.wav"

response=$(curl -s -X POST "$ENDPOINT" \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@$AUDIO_FILE" \
    -F "model=$MODEL" \
    -F "response_format=verbose_json")

text=$(echo "$response" | jq -r '.text')
duration=$(echo "$response" | jq -r '.duration')
processing_time=$(echo "$response" | jq -r '.processing_time')
rtfx=$(echo "$duration / $processing_time" | bc -l)

echo "Text: $text"
echo "RTFx: $rtfx"
```

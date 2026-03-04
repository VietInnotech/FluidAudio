# WebSocket Streaming ASR — Getting Started

FluidAudio includes a real-time WebSocket streaming endpoint compatible with Deepgram's protocol. Send raw PCM audio, receive transcription results with speaker timing and VAD-driven endpointing.

## Features

- **Deepgram-compatible protocol** — familiar URL param config, no JSON handshake
- **VAD-driven endpointing** — uses trained Silero VAD instead of energy thresholds
- **Streaming results** — `is_final`/`speech_final` flags for interim vs. final transcription
- **Per-utterance transcription** — speech detected independently per chunk
- **Binary audio input** — raw PCM (`pcm_s16le` or `pcm_f32le`) streamed directly
- **Multiple models** — Parakeet V2/V3, Qwen3, CTC Vietnamese
- **Configurable endpointing** — adjust silence duration to trigger utterance end

## Quick Start

### 1. Start the Server

```bash
cd /path/to/FluidAudio
swift build -c release --product fluidaudio-server
.build/release/fluidaudio-server
```

Server listens on `http://localhost:8080` (or configured host/port).

### 2. Connect and Send Audio

**URL format:**
```
ws://localhost:8080/v1/audio/stream
  ?model=parakeet-tdt-v3
  &language=en
  &encoding=pcm_s16le
  &interim_results=true
  &endpointing=700
```

**Query Parameters:**
- `model` — Model ID (`parakeet-tdt-v2`, `parakeet-tdt-v3`, `qwen3-f32`, `qwen3-int8`, `ctc-vi`). Default: `parakeet-tdt-v3`
- `language` — Language code (e.g., `en`, `fr`). Default: auto-detect
- `encoding` — Audio format (`pcm_s16le` or `pcm_f32le`). Default: `pcm_s16le`
- `interim_results` — Emit partial transcriptions while speaking. Default: `true`
- `endpointing` — Milliseconds of silence after speech that triggers utterance end. Default: `700`

### 3. Authentication (Optional)

If `FLUIDAUDIO_SERVER_API_KEY` is configured:

**Option A: Bearer header**
```bash
ws://localhost:8080/v1/audio/stream?model=parakeet-tdt-v3
Authorization: Bearer YOUR_API_KEY
```

**Option B: Query parameter**
```
ws://localhost:8080/v1/audio/stream?token=YOUR_API_KEY
```

## Protocol

### Connection

**Client connects:**
```
ws://localhost:8080/v1/audio/stream?model=parakeet-tdt-v3&language=en&interim_results=true&endpointing=700
```

**Server responds with `Metadata`:**
```json
{
  "type": "Metadata",
  "request_id": "a1b2c3d4-...",
  "model": "parakeet-tdt-v3"
}
```

### Sending Audio

**Send binary PCM frames immediately:**
```
[binary frame with PCM samples]
```

Audio must be:
- **16 kHz mono** (either `pcm_s16le` or `pcm_f32le` per `encoding` param)
- **Arbitrary chunk sizes** (VAD adapts internally)
- **Sequential** (no gaps or reordering)

### Server Messages

**Speech Started** — When VAD detects speech onset:
```json
{
  "type": "SpeechStarted",
  "timestamp": 2.5
}
```

**Results (interim)** — Streaming partial transcription while speaking:
```json
{
  "type": "Results",
  "transcript": "The quick brown",
  "start": 2.5,
  "duration": 1.2,
  "is_final": false,
  "speech_final": false,
  "channel": {
    "alternatives": [{"transcript": "The quick brown", "confidence": 1.0}]
  }
}
```

**Results (final)** — Complete utterance after silence timeout:
```json
{
  "type": "Results",
  "transcript": "The quick brown fox jumps over the lazy dog",
  "start": 2.5,
  "duration": 3.8,
  "is_final": true,
  "speech_final": true,
  "channel": {
    "alternatives": [{"transcript": "The quick brown fox jumps over the lazy dog", "confidence": 1.0}]
  }
}
```

**Utterance End** — Signals the end of a spoken utterance:
```json
{
  "type": "UtteranceEnd",
  "last_word_end": 6.3
}
```

**Error** — Connection or processing error:
```json
{
  "type": "Error",
  "description": "Transcription failed: ...",
  "variant": "transcription_error"
}
```

### Client Control Messages

Send as **text frames**:

**Finalize** — Force-flush current utterance without waiting for silence:
```json
{"type": "Finalize"}
```

**KeepAlive** — Prevent connection timeout during pauses:
```json
{"type": "KeepAlive"}
```

## Example: Python Client

```python
import asyncio
import json
import struct
import wave
import websockets

async def stream_audio(audio_file: str, model: str = "parakeet-tdt-v3"):
    """Stream audio file to FluidAudio streaming endpoint."""
    
    # Open audio file
    with wave.open(audio_file, 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        if wav.getsampwidth() != 2 or wav.getframerate() != 16000 or wav.getnchannels() != 1:
            raise ValueError("Audio must be 16kHz mono PCM S16LE")
    
    # Connect to streaming endpoint
    url = f"ws://localhost:8080/v1/audio/stream?model={model}&interim_results=true&endpointing=700"
    
    async with websockets.connect(url) as ws:
        # Receive metadata
        metadata = json.loads(await ws.recv())
        print(f"Connected: {metadata}")
        
        # Send audio in chunks (e.g., 320 samples = 20ms at 16kHz)
        chunk_size = 320 * 2  # 320 samples * 2 bytes per sample
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i+chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.02)  # 20ms realtime pacing
        
        # Finalize and read results
        await ws.send(json.dumps({"type": "Finalize"}))
        
        while True:
            try:
                msg = json.loads(await ws.recv())
                if msg["type"] == "Results":
                    print(f"[{msg['start']:.1f}s] {msg['transcript']}")
                elif msg["type"] == "UtteranceEnd":
                    print("Utterance ended")
                    break
            except asyncio.TimeoutError:
                break

# Run example
asyncio.run(stream_audio("audio.wav"))
```

## Example: JavaScript Client

```javascript
async function streamAudio(audioBuffer, model = "parakeet-tdt-v3") {
  const url = `ws://localhost:8080/v1/audio/stream?model=${model}&interim_results=true&endpointing=700`;
  const ws = new WebSocket(url);

  ws.onopen = () => {
    console.log("Connected");
    
    // Send audio in 320-sample chunks (20ms at 16kHz)
    const chunkSize = 320 * 2; // 320 samples * 2 bytes/sample for s16le
    let offset = 0;
    
    const sendChunks = () => {
      if (offset >= audioBuffer.byteLength) {
        ws.send(JSON.stringify({ type: "Finalize" }));
        return;
      }
      
      const chunk = audioBuffer.slice(offset, offset + chunkSize);
      ws.send(chunk);
      offset += chunkSize;
      
      setTimeout(sendChunks, 20); // 20ms realtime pacing
    };
    
    sendChunks();
  };

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) return;
    
    const msg = JSON.parse(event.data);
    
    if (msg.type === "Results") {
      console.log(`[${msg.start.toFixed(1)}s] ${msg.transcript}`);
    } else if (msg.type === "SpeechStarted") {
      console.log(`Speech started at ${msg.timestamp.toFixed(1)}s`);
    }
  };

  ws.onerror = (error) => console.error("WebSocket error:", error);
  ws.onclose = () => console.log("Disconnected");
}

// Load audio and stream
const audioContext = new AudioContext();
fetch("audio.wav")
  .then(r => r.arrayBuffer())
  .then(buf => audioContext.decodeAudioData(buf))
  .then(decoded => {
    // Convert AudioBuffer to PCM S16LE
    const pcm = convertToPcmS16LE(decoded.getChannelData(0), decoded.sampleRate);
    streamAudio(pcm.buffer);
  });
```

## Configuration

See `.env` file parameters in [ASR Server — Getting Started](./ServerGettingStarted.md):

- `FLUIDAUDIO_SERVER_HOST` — Listen address (default: `127.0.0.1`)
- `FLUIDAUDIO_SERVER_PORT` — Listen port (default: `8080`)
- `FLUIDAUDIO_SERVER_API_KEY` — Optional authentication key
- `FLUIDAUDIO_LOG_LEVEL` — Log verbosity (`debug`, `info`, `warning`, `error`)

## Tuning

### Endpointing Duration

Adjust the `endpointing` parameter (milliseconds of silence):
- **Short (300 ms):** Responsive but may cut off short pauses
- **Medium (700 ms):** Default, suitable for natural speech
- **Long (1000+ ms):** Waits longer, good for deliberate speakers

### Interim Results

Set `interim_results=false` to receive only final transcriptions (smaller payload, reduced CPU):
```
ws://localhost:8080/v1/audio/stream?interim_results=false
```

### Model Selection

- **Parakeet V3** — Default, fast, accurate across 25 European languages
- **Parakeet V2** — Legacy, use only if needed for consistency
- **Qwen3** — Multilingual, slightly slower than Parakeet
- **CTC Vietnamese** — Optimized for Vietnamese, narrow model

## Troubleshooting

### Connection Refused

Server not running or listening on wrong address/port. Check:
```bash
ps aux | grep fluidaudio-server
curl http://localhost:8080/openapi.json  # Should return API docs
```

### Audio Format Error

Verify audio is 16 kHz mono:
```bash
ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels audio.wav
```

Convert if needed:
```bash
ffmpeg -i audio.wav -acodec pcm_s16le -ac 1 -ar 16000 audio_16k.wav
```

### Slow Processing (RTFx < 1.0)

- Ensure release build: `swift build -c release`
- Use `-c release` flag, not debug
- Check CPU load: `top` should show single-threaded usage ~80–90%
- Reduce interim results frequency (set `interim_results=false` or wait between sends)

### Connection Timeout

Server unresponsive to new connections. Restart:
```bash
pkill -f fluidaudio-server
.build/release/fluidaudio-server
```

## Performance

Typical performance on M4 Pro:
- **Latency**: VAD endpointing + ASR processing ≈ 100–500 ms (depends on model & hardware)
- **Real-time factor (RTFx)**: 200–250x (processes 1 second of audio in 4–5 ms)
- **Memory**: ~500 MB for models + buffers
- **Concurrent streams**: Single stream at a time (queue subsequent connections)

## Known Limitations

1. **Single-threaded transcription** — only one stream is processed at a time; others wait
2. **Models load on first use** — first request is slower (automatic download + compilation)
3. **No authentication by default** — use `FLUIDAUDIO_SERVER_API_KEY` in production
4. **Macbook only** — server requires macOS 14+ with Apple Silicon

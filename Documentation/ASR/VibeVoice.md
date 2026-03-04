# VibeVoice-ASR Integration

VibeVoice-ASR is Microsoft's unified speech-to-text model that jointly performs:
- **Automatic Speech Recognition** (ASR)
- **Speaker Diarization** (who spoke when)
- **Timestamping** (precise start/end times)

It processes up to 60 minutes of audio in a single pass, outputting structured JSON with speaker-attributed, timestamped transcription segments.

## Model Details

| Property | Value |
|---|---|
| **Architecture** | Acoustic/Semantic encoders → Qwen2.5-7B decoder |
| **Parameters** | ~9B (17.3 GB safetensors) |
| **Audio** | 24 kHz mono |
| **Languages** | 50+ (auto-detected) |
| **Max Duration** | 60 minutes |
| **License** | MIT |
| **Source** | [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) |

## Quick Start

```swift
import FluidAudio

let manager = VibeVoiceAsrManager()

// Download models (one-time, ~14 GB for FP16 or ~5 GB for INT4)
let modelDir = try await VibeVoiceAsrModels.download(variant: .f32)

// Load models
let models = try await VibeVoiceAsrModels.load(from: modelDir)
try await manager.loadModels(models)

// Transcribe audio (must be 24kHz mono Float32)
let result = try await manager.transcribe(audioSamples: samples)

// Access structured results
for segment in result.segments {
    print("[\(segment.startTime)-\(segment.endTime)] Speaker \(segment.speakerId): \(segment.content)")
}

// Or get plain text
print(result.plainText)
```

### Model Status

✅ **CoreML models are now available** at `FluidInference/vibevoice-asr-coreml` (March 2026)
- `.f32` variant: Full precision (14 GB) — best quality
- `.int4` variant: INT4 quantized (3.5 GB) — faster, smaller

## Model Variants

| Variant | Decoder Size | Total Size | Quality | Use Case |
|---|---|---|---|---|
| `.f32` | ~14 GB | ~16 GB | Best | Mac Studio / Mac Pro |
| `.int4` | ~3.5 GB | ~5 GB | Good | MacBook / iPad Pro |

```swift
// Download INT4 quantized (recommended for most devices)
let modelDir = try await VibeVoiceAsrModels.download(variant: .int4)
```

## Output Format

VibeVoice-ASR produces structured JSON output that is automatically parsed into Swift types:

```swift
// Each segment contains:
struct VibeVoiceTranscriptionSegment {
    let startTime: String    // e.g., "0.00"
    let endTime: String      // e.g., "3.45"
    let speakerId: String    // e.g., "Speaker 1"
    let content: String      // e.g., "Hello, welcome to the meeting."
}

// The result provides computed properties:
let result: VibeVoiceTranscriptionResult
result.segments         // [VibeVoiceTranscriptionSegment]
result.speakerCount     // Number of unique speakers
result.totalDuration    // Total duration of all segments  
result.plainText        // Concatenated transcription text
```

## Providing Context

You can pass context to improve transcription accuracy for domain-specific content:

```swift
let result = try await manager.transcribe(
    audioSamples: samples,
    context: "Technical discussion about CoreML, FluidAudio, and speaker diarization"
)
```

Context is useful for:
- Domain terminology and jargon
- Speaker names (if known in advance)
- Expected language or topic

## Audio Preparation

VibeVoice-ASR requires **24 kHz mono Float32** audio. Use `AudioConverter` if your source is different:

```swift
let converter = AudioConverter()
let samples = try await converter.convertToFloat32Array(
    from: audioFileURL,
    targetSampleRate: 24000
)
```

## Architecture

### Pipeline

```
Audio (24kHz) ──┬── Acoustic Encoder ──┐
                │                       ├── Feature Merge ── Qwen2.5-7B Decoder ── JSON
                └── Semantic Encoder ──┘
```

### Components

1. **Acoustic Tokenizer Encoder**: VAE with [8,5,5,4,2,2] strided convolutions (3200x compression). Outputs 64-dim latent features.

2. **Semantic Tokenizer Encoder**: Similar architecture, outputs 128-dim semantic features.

3. **Audio Projector**: Projects acoustic + semantic features into the LLM's 3584-dim hidden space.

4. **Qwen2.5-7B Decoder**: 28-layer transformer with:
   - 28 attention heads, 4 KV heads (GQA)
   - 3584 hidden size, 18944 intermediate size
   - Standard RoPE (θ=1,000,000)
   - Stateful KV cache for autoregressive generation

### CoreML Models

| File | Description | Size |
|---|---|---|
| `vibevoice_acoustic_encoder.mlmodelc` | Acoustic VAE encoder | ~50 MB |
| `vibevoice_semantic_encoder.mlmodelc` | Semantic encoder | ~50 MB |
| `vibevoice_decoder_stateful.mlmodelc` | Fused Qwen2.5-7B + lm_head | ~14 GB (FP16) |
| `vibevoice_embeddings.bin` | Float16 embedding matrix | ~1 GB |
| `vocab.json` | Tokenizer vocabulary (152,064 tokens) | ~5 MB |

## Converting Models

Use the conversion script to create CoreML models from the HuggingFace checkpoint:

```bash
cd Tools

# Convert all components (requires ~40 GB RAM)
uv run convert_vibevoice_asr_to_coreml.py

# Convert only the decoder
uv run convert_vibevoice_asr_to_coreml.py --components decoder

# INT4 quantization (post-conversion, use coremltools)
uv run convert_vibevoice_asr_to_coreml.py --output-dir ./build/vibevoice-asr-int4
```

### Requirements

- Python 3.10+
- `torch`, `transformers`, `coremltools>=8.0`, `safetensors`
- ~40 GB RAM for conversion
- Apple Silicon Mac recommended

## Comparison with Other ASR Models

| Feature | VibeVoice | Parakeet TDT | Qwen3-ASR | Whisper |
|---|---|---|---|---|
| Speaker diarization | ✅ Built-in | ❌ | ❌ | ❌ |
| Timestamps | ✅ Built-in | ✅ Token-level | ⚠️ Sentence-level | ✅ Word-level |
| Max duration | 60 min | ~15s chunks | ~30s chunks | ~30s chunks |
| Languages | 50+ | 25 | 13 | 99 |
| Model size | 9B | 0.6B | 0.6-1.7B | 0.04-1.5B |
| On-device speed | Slower | Fast (210x RTF) | Medium | Medium |
| Use case | Meetings, interviews | Real-time streaming | Multilingual | General purpose |

## Limitations

- **Model size**: 9B parameters requires Apple Silicon with ≥16 GB unified memory (INT4) or ≥32 GB (FP16)
- **Not real-time**: Processing happens after audio capture; not suitable for live streaming
- **macOS/iOS only**: CoreML models require Apple platforms (macOS 15+, iOS 18+)
- **First-time download**: Large model download (~14 GB for FP16, ~5 GB for INT4)

## References

- [VibeVoice Paper](https://arxiv.org/abs/2601.18184)
- [VibeVoice-ASR on HuggingFace](https://huggingface.co/microsoft/VibeVoice-ASR)
- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)

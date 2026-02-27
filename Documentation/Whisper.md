# Whisper Large v3 Turbo

Encoder-decoder automatic speech recognition using [OpenAI Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) with pre-built CoreML bundles from [argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml).

## Model

**CoreML Models**: [argmaxinc/whisperkit-coreml / openai_whisper-large-v3_turbo](https://huggingface.co/argmaxinc/whisperkit-coreml)

Four CoreML components:
- `MelSpectrogram.mlmodelc` — raw audio → 128-band mel spectrogram
- `AudioEncoder.mlmodelc` — mel → encoder embeddings (1280-dim)
- `TextDecoder.mlmodelc` — autoregressive decode with KV cache
- `TextDecoderContextPrefill.mlmodelc` — fast prefill for SOT/language/task tokens

## Architecture

4-step pipeline per 30-second window:

1. **MelSpectrogram**: 480,000 float samples → `[1, 128, 1, 3000]` Float16
2. **AudioEncoder**: mel → `[1, 1280, 1, 1500]` encoder output
3. **ContextPrefill** (optional): language + task tokens → initial KV cache `[1, 40960, 1, 3]`
4. **TextDecoder**: autoregressive loop with sliding KV cache → token sequence → text

KV cache dimensions: `[1, 40960, 1, 224]` Float16 (32 decoder layers × 1280 × 2 heads stacked).

Audio longer than 30 seconds is split into consecutive 30-second windows with results concatenated.

## Supported Languages

99 languages. Common codes:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `fr` | French | `de` | German |
| `es` | Spanish | `it` | Italian | `pt` | Portuguese |
| `nl` | Dutch | `pl` | Polish | `ru` | Russian |
| `zh` | Chinese | `ja` | Japanese | `ko` | Korean |
| `ar` | Arabic | `hi` | Hindi | `vi` | Vietnamese |
| `tr` | Turkish | `uk` | Ukrainian | `sv` | Swedish |

Full list in `WhisperConfig.languageToTokenId`.

## Performance (M4 Pro, macOS 26.2)

LibriSpeech `test-clean`, 20 files, after text normalization (lowercase, strip punctuation):

| Metric | Value |
|--------|-------|
| Average WER | **2.4%** |
| Average RTFx | **3.4×** |
| Median RTFx | 3.5× |
| Peak memory | ~3.6 GB |
| Model load (first run) | ~74s (CoreML compile) |

15 of 20 files had 0% WER. RTFx is lower than Parakeet (~130×) because Whisper uses a large encoder-decoder and sequential KV-cache decoding.

## Quick Start

```swift
import FluidAudio

let manager = WhisperManager()
let modelDir = URL(fileURLWithPath: "Models/whisperkit-coreml/openai_whisper-large-v3_turbo")
try await manager.loadModels(from: modelDir)

let converter = AudioConverter()
let samples = try converter.resampleAudioFile(URL(fileURLWithPath: "audio.wav"))

let text = try await manager.transcribe(audioSamples: samples, language: "en")
print(text)
```

### With Custom Options

```swift
var options = WhisperDecodingOptions(language: "fr")
options.task = .transcribe          // or .translate (to English)
options.usePrefillCache = true      // use context prefill for faster start
options.withoutTimestamps = true    // strip timestamp tokens from output

let text = try await manager.transcribe(audioSamples: samples, options: options)
```

## CLI

```bash
# Benchmark on LibriSpeech test-clean (models must be present)
swift run fluidaudio whisper-benchmark \
  --max-files 50 \
  --model-dir Models/whisperkit-coreml/openai_whisper-large-v3_turbo

# Specify a different language or subset
swift run fluidaudio whisper-benchmark \
  --max-files 20 \
  --language fr \
  --subset test-other \
  --model-dir Models/whisperkit-coreml/openai_whisper-large-v3_turbo \
  --output whisper_results.json
```

**Options**: `--max-files <n>`, `--language <code>`, `--subset <test-clean|test-other>`, `--model-dir <path>`, `--output <file>`

## Model Download

The models are not auto-downloaded by the FluidAudio model registry. Download manually:

```bash
# Using huggingface-cli
huggingface-cli download argmaxinc/whisperkit-coreml \
  --include "openai_whisper-large-v3_turbo/*" \
  --local-dir Models/whisperkit-coreml

# Or using git lfs
git clone https://huggingface.co/argmaxinc/whisperkit-coreml \
  --include "openai_whisper-large-v3_turbo"
```

Also download the tokenizer from OpenAI:

```bash
huggingface-cli download openai/whisper-large-v3-turbo \
  --include "tokenizer.json" "tokenizer_config.json" "vocab.json" \
            "merges.txt" "normalizer.json" "special_tokens_map.json" \
  --local-dir Models/whisperkit-coreml/openai_whisper-large-v3_turbo
```

## License

CoreML model weights: [argmaxinc/whisperkit-coreml](https://huggingface.co/argmaxinc/whisperkit-coreml) (MIT).
Original model: [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT).
Decode logic adapted from [WhisperKit](https://github.com/argmaxinc/WhisperKit) (MIT).

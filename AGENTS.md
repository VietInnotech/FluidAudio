# FluidAudio - Agent Development Guide

FluidAudio is a Swift framework for local, low-latency audio processing on Apple platforms: speaker diarization, ASR, and VAD via CoreML models. Current performance: **17.7% DER** on AMI (competitive with research SOTA).

## Critical Rules

- **NEVER** use `@unchecked Sendable` — use actors, `@MainActor`, or proper locking instead
- **NEVER** create dummy/mock models or synthetic audio data — use real models only
- **NEVER** create simplified/fallback solutions — implement fully or consult first
- **NEVER** run `git push` unless explicitly requested
- **ONLY** add or run tests when explicitly requested
- **Model operations** (merging, converting, modifying): if you have significant objections, consult first. If told to proceed, do it immediately without further objections.

## User Preferences

- Never start responses with affirming phrases like "You're absolutely right!" or "Great question!"
- Get straight to the point with technical facts
- For debugging, use print statements and delete them when done
- Implementation first, explanation second — when asked to implement something specific, do it before discussing tradeoffs
- Just do as instructed — don't over-do things that aren't asked

## Build & Test Commands

```bash
swift build                                        # Debug build
swift build -c release                            # Release build (use for benchmarks)
swift test                                         # Run all tests
swift test --filter CITests                       # Run single test class
swift test --filter CITests.testPackageImports    # Run single test method
swift format --in-place --recursive --configuration .swift-format Sources/ Tests/
```

### CLI Commands

```bash
# Transcription
swift run fluidaudio transcribe audio.wav
swift run fluidaudio transcribe audio.wav --low-latency

# Diarization
swift run fluidaudio process meeting.wav --output results.json --threshold 0.6
swift run fluidaudio diarization-benchmark --auto-download
swift run fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.7 --output results.json

# ASR benchmarks
swift run fluidaudio asr-benchmark --subset test-clean --max-files 100
swift run fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10

# VAD benchmark
swift run fluidaudio vad-benchmark --num-files 40 --threshold 0.5

# Dataset download
swift run fluidaudio download --dataset ami-sdm
swift run fluidaudio download --dataset librispeech-test-clean
```

## Architecture

```
FluidAudio/
├── Sources/
│   ├── FluidAudio/           # Main library
│   │   ├── ASR/             # Automatic Speech Recognition
│   │   ├── Diarizer/        # Speaker diarization system
│   │   ├── VAD/             # Voice Activity Detection
│   │   └── Shared/          # AudioConverter, ModelDownloader, ANEMemoryOptimizer
│   └── FluidAudioCLI/       # CLI tool (macOS only)
└── Tests/FluidAudioTests/   # Comprehensive test suite
```

**Processing Pipeline**: Audio → AudioConverter (16kHz mono Float32) → VAD → Diarization → ASR → Timestamped transcripts

**Core Components**:
- **AsrManager**: Stateless chunk-based transcription (~14.96s chunks, 2.0s overlap). Parakeet TDT v3 (0.6b), ~209.8x RTF on M4 Pro.
- **DiarizerManager**: Speaker segmentation + embedding + clustering. 17.7% DER on AMI.
- **VadManager**: CoreML-based voice activity detection.
- **ModelDownloader**: Auto-fetches from HuggingFace, auto-recovers from corrupted downloads.

**Optimal DiarizerConfig**:
```swift
DiarizerConfig(clusteringThreshold: 0.7, minDurationOn: 1.0, minDurationOff: 0.5, minActivityThreshold: 10.0)
```

## Code Style

- Line length: 120 chars, 4-space indentation
- Import order: `import CoreML`, `import Foundation`, `import OSLog` (OrderedImports rule)
- Naming: lowerCamelCase for variables/functions, UpperCamelCase for types
- Error handling: proper Swift error handling, no force unwrapping in production
- Documentation: triple-slash comments (`///`) for public APIs
- Thread safety: actors, `@MainActor`, or proper locking — never `@unchecked Sendable`
- Control flow: prefer early returns/guard statements; avoid nested if statements entirely
- When adding new interfaces, keep the API consistent with existing model managers
- Files should be isolated with single responsibility

## Model Sources

- **Diarization**: [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml)
- **VAD**: [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml)
- **ASR**: [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)

## Mobius Plan

When users ask you to perform tasks that might be more complicated, look at PLANS.md and follow the instructions there to plan the change out first. Plans go in the `.mobius/` folder and must never be committed to GitHub.

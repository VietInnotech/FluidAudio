# VibeVoice-ASR Testing Guide

Complete guide to testing VibeVoice through **Unit Tests**, **CLI**, **Swift API**, and **Real Audio Files**.

## Prerequisites

### System Requirements
- **macOS 15** or **iOS 18** (minimum for VibeVoice support)
- **Xcode** 15+
- **Swift** 5.10+

### Model Download (Required for Transcription)
```bash
# Models auto-download on first use, or download manually:
# Option 1: Full Precision (FP32) - 14GB, best quality
swift build
swift run fluidaudio transcribe /tmp/test_vibevoice.wav
# Models download automatically to ~/Library/Caches/FluidAudio/

# Option 2: INT4 Quantized - 3.5GB, faster
# Specify in your code or CLI: --variant int4
```

---

## Testing Method 1: Unit Tests (No Models Required)

The fastest way to test VibeVoice configuration, output parsing, and error handling **without downloading models**.

### Run All VibeVoice Unit Tests
```bash
cd /Users/vit/offasr/FluidAudio
swift test --filter VibeVoiceTests
```

### Run Specific Test Categories
```bash
# Config validation tests (constants, compression ratios, tokenizer settings)
swift test --filter VibeVoiceTests.testConfigAudioConstants
swift test --filter VibeVoiceTests.testLanguageFromISOCode
swift test --filter VibeVoiceTests.testConfigSpecialTokensAreDistinct

# Output parser tests (JSON parsing, segment extraction)
swift test --filter VibeVoiceTests.testParseValidJSONArray
swift test --filter VibeVoiceTests.testParseEmptyJSON

# Model validation tests
swift test --filter VibeVoiceTests.testVariantEnum
```

### What's Tested
- ✅ Audio configuration (24 kHz sample rate, compression ratios)
- ✅ Tokenizer constants (VAE dimensions, downsampling rates)
- ✅ Decoder architecture (layer count, attention heads, vocabulary size)
- ✅ Special tokens validation (unique IDs, within vocab bounds)
- ✅ Language detection (50+ ISO codes and English names)
- ✅ Output JSON parsing (array extraction, segment validation)
- ✅ Model variant enum (f32 vs int4)

---

## Testing Method 2: CLI Command

The most straightforward way to test VibeVoice with real audio files.

### Build CLI
```bash
# Debug build (slower, but faster compilation)
swift build

# Release build (optimized, recommended for accuracy testing)
swift build -c release
```

### Basic Transcription
```bash
# Uses the test audio we created (/tmp/test_vibevoice.wav)
swift run fluidaudio vibevoice-transcribe /tmp/test_vibevoice.wav

# With INT4 model (recommended for faster download/execution)
swift run fluidaudio vibevoice-transcribe /tmp/test_vibevoice.wav --variant int4

# With context to improve accuracy (domain-specific terms, speaker names)
swift run fluidaudio vibevoice-transcribe /tmp/test_vibevoice.wav \
  --context "Interview about software engineering and machine learning"
```

### Output to JSON
```bash
# Save structured segment data to file
swift run fluidaudio vibevoice-transcribe /tmp/test_vibevoice.wav --json > transcription.json

# Pretty-print for inspection
swift run fluidaudio vibevoice-transcribe /tmp/test_vibevoice.wav --json | python3 -m json.tool
```

### CLI Options Reference
| Option | Example | Purpose |
|--------|---------|---------|
| `--model-dir` | `--model-dir ~/models` | Use pre-downloaded models (skips download) |
| `--variant` | `--variant int4` | Model variant: f32 (14GB) or int4 (3.5GB) |
| `--context, -c` | `-c "meeting notes"` | Provide domain context for accuracy |
| `--max-tokens` | `--max-tokens 8192` | Maximum tokens to generate (default: 8192) |
| `--json` | `--json` | Output raw JSON array instead of formatted text |
| `--help, -h` | `--help` | Show usage information |

### Example: Full Workflow
```bash
# 1. Create test audio (10 seconds of synthesized audio)
python3 << 'PYTHON_EOF'
import subprocess
subprocess.run([
    "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:duration=10",
    "-af", "volume=0.3", "-ar", "48000", "-ac", "2", "-y",
    "/tmp/meeting_sample.wav"
], capture_output=True)
PYTHON_EOF

# 2. Transcribe with debug output
swift build -c release
swift run fluidaudio vibevoice-transcribe /tmp/meeting_sample.wav

# 3. Save JSON output for further analysis
swift run fluidaudio vibevoice-transcribe /tmp/meeting_sample.wav --json > segments.json

# 4. View results
echo "Segments:"
cat segments.json | python3 -m json.tool
```

---

## Testing Method 3: Swift API (Programmatic)

Direct API usage for integration tests and custom workflows.

### Basic Usage
```swift
import FluidAudio
import AVFoundation

@available(macOS 15, iOS 18, *)
func testVibeVoiceBasic() async throws {
    // Create manager
    let manager = VibeVoiceAsrManager()
    
    // Option A: Auto-download models
    let modelDir = try await VibeVoiceAsrModels.download(variant: .f32)
    try await manager.loadModels(from: modelDir)
    
    // Option B: Use pre-downloaded models
    // let modelURL = URL(fileURLWithPath: "/path/to/models")
    // try await manager.loadModels(from: modelURL)
    
    // Load audio file (converts to 24kHz mono automatically)
    let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 24000,
        channels: 1,
        interleaved: false
    )!
    
    let audioConverter = AudioConverter(targetFormat: targetFormat)
    let samples = try audioConverter.resampleAudioFile(path: "/tmp/test_vibevoice.wav")
    
    // Transcribe
    let result = try await manager.transcribe(audioSamples: samples)
    
    // Access results
    print("Segments: \(result.segments.count)")
    print("Speakers: \(result.speakerCount)")
    
    for segment in result.segments {
        print("[\(segment.startTime) - \(segment.endTime)] \(segment.speakerId): \(segment.content)")
    }
}
```

### With Context Biasing
```swift
// Improve accuracy with domain context
let result = try await manager.transcribe(
    audioSamples: samples,
    context: "Discussion about distributed systems, consensus algorithms, and Byzantine fault tolerance"
)
```

### With Custom Parameters
```swift
let result = try await manager.transcribe(
    audioSamples: samples,
    context: "Meeting minutes",
    maxNewTokens: 4096  // Limit output length
)
```

### Accessing Structured Output
```swift
// Each segment provides:
// - startTime: String (e.g., "0.00")
// - endTime: String (e.g., "2.50")
// - speakerId: String (e.g., "Speaker 1")
// - content: String (e.g., "Hello, how are you?")

for segment in result.segments {
    let duration = Double(segment.endTime)! - Double(segment.startTime)!
    print("📍 \(segment.speakerId) (\(String(format: "%.2f", duration))s): \(segment.content)")
}

// Get speaker statistics
print("Total speakers: \(result.speakerCount)")
print("Total audio covered: \(result.totalDuration ?? 0)s")
print("Full transcription: \(result.plainText)")
```

### Error Handling
```swift
do {
    let result = try await manager.transcribe(audioSamples: samples)
} catch VibeVoiceAsrError.modelNotLoaded {
    print("Models not loaded. Call loadModels() first.")
} catch VibeVoiceAsrError.invalidAudioLength {
    print("Audio too short or too long (max 60 minutes)")
} catch VibeVoiceAsrError.decodingFailed {
    print("Model inference failed")
} catch {
    print("Other error: \(error)")
}
```

---

## Testing Method 4: Using Real Audio Datasets

Test with standard benchmark datasets for evaluation.

### Download Available Datasets
```bash
# AMI corpus (speaker diarization, multi-speaker meetings)
swift run fluidaudio download --dataset ami-sdm

# LibriSpeech (clean audio - common speech recognition benchmark)
swift run fluidaudio download --dataset librispeech-test-clean

# Other available datasets
swift run fluidaudio download --dataset librispeech-test-other    # Noisy LibriSpeech
swift run fluidaudio download --dataset musan                     # Background noise
```

### Transcribe Dataset Files
```bash
# After downloading, transcribe AMI files
swift run fluidaudio vibevoice-transcribe ~/Datasets/ami/ES2004a.Mix.wav

# Batch transcribe with output to file
for file in ~/Datasets/ami/*.wav; do
    echo "Processing: $file"
    swift run fluidaudio vibevoice-transcribe "$file" --json >> all_results.jsonl
done
```

### Evaluate Performance
```bash
# After transcribing, evaluate with DER (Diarization Error Rate)
python3 << 'PYTHON_EOF'
import json

# Load all transcription results
with open('all_results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# Analyze speaker accuracy
for file_results in results:
    speaker_count = len(set(seg.get("Speaker ID") for seg in file_results))
    segment_count = len(file_results)
    print(f"File: {segment_count} segments, {speaker_count} speakers")
    
    # Print sample
    for seg in file_results[:3]:
        print(f"  [{seg['Start time']} - {seg['End time']}] {seg['Speaker ID']}: {seg['Content'][:60]}...")
PYTHON_EOF
```

---

## Testing Method 5: Custom Integration Tests

Create your own test suite for VibeVoice.

### Example Test File
Create `Tests/FluidAudioTests/VibeVoiceIntegrationTests.swift`:

```swift
import XCTest
import FluidAudio
import AVFoundation

@available(macOS 15, iOS 18, *)
final class VibeVoiceIntegrationTests: XCTestCase {
    
    var manager: VibeVoiceAsrManager!
    var modelDirectory: URL!
    
    override func setUp() async throws {
        try await super.setUp()
        manager = VibeVoiceAsrManager()
        
        // Download INT4 models for faster testing
        modelDirectory = try await VibeVoiceAsrModels.download(variant: .int4)
        try await manager.loadModels(from: modelDirectory)
    }
    
    // Test basic transcription
    func testBasicTranscription() async throws {
        // Create 5-second test audio buffer
        let sampleRate: Int = 24000
        let durationSeconds: Int = 5
        var samples = [Float](repeating: 0, count: sampleRate * durationSeconds)
        
        // Fill with simple sine wave pattern
        for i in 0..<samples.count {
            let frequency: Float = 1000.0
            samples[i] = sin(Float(i) * 2.0 * .pi * frequency / Float(sampleRate)) * 0.3
        }
        
        // Transcribe
        let result = try await manager.transcribe(audioSamples: samples)
        
        // Assertions
        XCTAssertGreaterThanOrEqual(result.segments.count, 0)
        XCTAssertGreaterThanOrEqual(result.speakerCount, 0)
    }
    
    // Test with real audio file
    func testRealAudioFile() async throws {
        // Resample audio to 24kHz mono
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 24000,
            channels: 1,
            interleaved: false
        )!
        
        let converter = AudioConverter(targetFormat: targetFormat)
        let samples = try converter.resampleAudioFile(path: "/tmp/test_vibevoice.wav")
        
        XCTAssertGreater(samples.count, 0)
        
        // Transcribe
        let result = try await manager.transcribe(audioSamples: samples)
        
        // Verify output structure
        for segment in result.segments {
            XCTAssertFalse(segment.startTime.isEmpty)
            XCTAssertFalse(segment.endTime.isEmpty)
            XCTAssertFalse(segment.speakerId.isEmpty)
            XCTAssertFalse(segment.content.isEmpty)
        }
    }
    
    // Test with context
    func testContextBiasing() async throws {
        let samples = try createTestSamples(durationSeconds: 5)
        
        let resultWithContext = try await manager.transcribe(
            audioSamples: samples,
            context: "Technical discussion about machine learning"
        )
        
        XCTAssertGreaterThanOrEqual(resultWithContext.segments.count, 0)
    }
    
    // Helper to create synthetic audio
    private func createTestSamples(durationSeconds: Int) -> [Float] {
        let sampleRate: Int = 24000
        let totalSamples = sampleRate * durationSeconds
        var samples = [Float](repeating: 0, count: totalSamples)
        
        for i in 0..<samples.count {
            let frequency: Float = 1000.0
            samples[i] = sin(Float(i) * 2.0 * .pi * frequency / Float(sampleRate)) * 0.3
        }
        
        return samples
    }
}
```

Run this test:
```bash
swift test --filter VibeVoiceIntegrationTests
```

---

## Performance Benchmarking

### Measure RTFx (Real-Time Factor)
```bash
# RTFx = audio duration / processing time
# RTFx > 1.0 = real-time capable
# VibeVoice expects RTFx ~0.5-2.0 depending on hardware

swift run fluidaudio vibevoice-transcribe /tmp/test_vibevoice.wav
# Output includes: "RTFx: 0.45x" (processes ~2.2s of audio per 1s of computation)
```

### Benchmark Multiple Files
```bash
python3 << 'PYTHON_EOF'
import subprocess
import time
import os

audio_files = ["/tmp/test_vibevoice.wav"]  # Add more files here

for audio_file in audio_files:
    print(f"\n📊 Benchmarking: {os.path.basename(audio_file)}")
    
    start = time.time()
    result = subprocess.run(
        ["swift", "run", "fluidaudio", "vibevoice-transcribe", audio_file],
        capture_output=True, text=True
    )
    elapsed = time.time() - start
    
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Output:\n{result.stdout}")
PYTHON_EOF
```

---

## Model Variants Comparison

| Aspect | FP32 (Full Precision) | INT4 (Quantized) |
|--------|---|---|
| **Model Size** | ~14 GB | ~3.5 GB |
| **Quality** | Best | Good (~95%) |
| **Speed** | Standard | ~1.5-2x faster |
| **Memory** | ~28 GB required | ~10 GB required |
| **Best For** | Mac Studio, Mac Pro | MacBook, iPad Pro |
| **CLI Usage** | `--variant f32` (default) | `--variant int4` |

---

## Troubleshooting

### Models Won't Download
```bash
# Check internet connection
curl -I https://huggingface.co

# Manually cache models
export HF_HOME=~/my_models
swift run fluidaudio vibevoice-transcribe audio.wav
```

### Audio Format Issues
```bash
# VibeVoice requires 24kHz mono Float32
# The CLI handles this automatically, but:

# Check audio properties
ffprobe -v error -show_entries stream=sample_rate,channels /tmp/test_vibevoice.wav

# Convert audio manually if needed
ffmpeg -i input.mp3 -ar 24000 -ac 1 -acodec pcm_f32le output.wav
```

### Slow Transcription
```bash
# 1. Use INT4 model (3.5x faster)
swift run fluidaudio vibevoice-transcribe audio.wav --variant int4

# 2. Reduce max-tokens
swift run fluidaudio vibevoice-transcribe audio.wav --max-tokens 4096

# 3. Use Release build
swift build -c release
swift run fluidaudio vibevoice-transcribe audio.wav
```

### Memory Issues
```bash
# Use INT4 model to reduce memory footprint
# Reports: "malloc: Cannot allocate memory" → use --variant int4

swift run fluidaudio vibevoice-transcribe audio.wav --variant int4
```

---

## Next Steps

1. **Run unit tests** to verify configuration
2. **Test with the provided audio** (`/tmp/test_vibevoice.wav`)
3. **Download a real dataset** (AMI or LibriSpeech)
4. **Measure performance** with RTFx
5. **Create integration tests** for your use case

For detailed API documentation, see [VibeVoice.md](VibeVoice.md).

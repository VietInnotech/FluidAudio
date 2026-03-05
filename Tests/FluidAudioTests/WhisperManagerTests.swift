import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

// MARK: - Whisper Unit Tests (no models required)

/// Tests for Whisper configuration, error types, and manager state —
/// none of these require model files on disk.
final class WhisperConfigTests: XCTestCase {

    // MARK: kvCacheEmbedDim must match real turbo (4 decoder layers)

    func testKvCacheEmbedDimMatchesRealTurbo() {
        // Real Whisper Large v3 Turbo has 4 decoder layers × 1280 dim × 2 (K+V) = 5120
        // The 32-layer large-v3 would be 40960 — this guards against regression.
        XCTAssertEqual(WhisperConfig.kvCacheEmbedDim, 5120)
    }

    func testModelDimensionsAreConsistent() {
        XCTAssertEqual(WhisperConfig.dModel, 1280)
        XCTAssertEqual(WhisperConfig.decoderAttentionHeads, 20)
        XCTAssertEqual(WhisperConfig.numMelBins, 128)
        XCTAssertEqual(WhisperConfig.maxSourcePositions, 1500)
        XCTAssertEqual(WhisperConfig.maxTargetPositions, 448)
        XCTAssertEqual(WhisperConfig.maxTokenContext, 224)
        XCTAssertEqual(WhisperConfig.sampleRate, 16000)
        XCTAssertEqual(WhisperConfig.windowSamples, 480_000)
    }

    func testSecondsPerTimeTokenIsCorrect() {
        // Whisper timestamps step at 0.02s per token
        XCTAssertEqual(WhisperConfig.secondsPerTimeToken, 0.02, accuracy: 0.0001)
        // 1500 max source positions × 0.02 = 30s window
        let windowSeconds = Float(WhisperConfig.maxSourcePositions) * WhisperConfig.secondsPerTimeToken
        XCTAssertEqual(windowSeconds, 30.0, accuracy: 0.01)
    }

    // MARK: Token IDs

    func testSpecialTokenIDsAreCorrect() {
        XCTAssertEqual(WhisperConfig.Tokens.endOfText, 50257)
        XCTAssertEqual(WhisperConfig.Tokens.startOfTranscript, 50258)
        XCTAssertEqual(WhisperConfig.Tokens.english, 50259)
        XCTAssertEqual(WhisperConfig.Tokens.transcribe, 50360)
        XCTAssertEqual(WhisperConfig.Tokens.translate, 50359)
        XCTAssertEqual(WhisperConfig.Tokens.noTimestamps, 50364)
        XCTAssertEqual(WhisperConfig.Tokens.noSpeech, 50362)
    }

    func testVocabSizeCoverageForSpecialTokens() {
        // All special tokens must be within vocabulary bounds
        let specialTokens = [
            WhisperConfig.Tokens.endOfText,
            WhisperConfig.Tokens.startOfTranscript,
            WhisperConfig.Tokens.english,
            WhisperConfig.Tokens.transcribe,
            WhisperConfig.Tokens.translate,
            WhisperConfig.Tokens.noTimestamps,
            WhisperConfig.Tokens.noSpeech,
        ]
        for token in specialTokens {
            XCTAssertLessThan(token, WhisperConfig.vocabSize, "Special token \(token) exceeds vocab size")
        }
    }

    // MARK: Language lookup

    func testLanguageLookupKnownLanguages() {
        XCTAssertEqual(WhisperConfig.languageTokenId(for: "en"), 50259)
        XCTAssertEqual(WhisperConfig.languageTokenId(for: "fr"), 50265)
        XCTAssertEqual(WhisperConfig.languageTokenId(for: "de"), 50261)
        XCTAssertEqual(WhisperConfig.languageTokenId(for: "zh"), 50260)
        XCTAssertEqual(WhisperConfig.languageTokenId(for: "vi"), 50278)
    }

    func testLanguageLookupNilDefaultsToEnglish() {
        XCTAssertEqual(WhisperConfig.languageTokenId(for: nil), WhisperConfig.Tokens.english)
    }

    func testLanguageLookupUnknownDefaultsToEnglish() {
        XCTAssertEqual(WhisperConfig.languageTokenId(for: "xx"), WhisperConfig.Tokens.english)
    }

    func testAllLanguageMappingTokensInVocab() {
        for (code, token) in WhisperConfig.languageToTokenId {
            XCTAssertLessThan(token, WhisperConfig.vocabSize, "Language \(code) token \(token) exceeds vocab")
        }
    }

    // MARK: Suppress tokens

    func testSuppressTokensIsNotEmpty() {
        XCTAssertFalse(WhisperConfig.suppressTokens.isEmpty)
    }

    func testSuppressTokensWithinVocab() {
        for token in WhisperConfig.suppressTokens {
            XCTAssertLessThan(token, WhisperConfig.vocabSize, "Suppress token \(token) out of range")
        }
    }
}

// MARK: - WhisperError Tests

final class WhisperErrorTests: XCTestCase {

    func testAllErrorCasesHaveDescriptions() {
        let errors: [WhisperError] = [
            .modelLoadFailed("path"),
            .melExtractionFailed("reason"),
            .encodingFailed("reason"),
            .decodingFailed("reason"),
            .tokenizerError("reason"),
            .audioTooShort,
        ]
        for error in errors {
            XCTAssertFalse((error.errorDescription ?? "").isEmpty, "\(error) has empty description")
        }
    }

    func testErrorDescriptionsContainMessage() {
        let msg = "custom detail"
        XCTAssertTrue(WhisperError.modelLoadFailed(msg).errorDescription?.contains(msg) == true)
        XCTAssertTrue(WhisperError.decodingFailed(msg).errorDescription?.contains(msg) == true)
        XCTAssertTrue(WhisperError.tokenizerError(msg).errorDescription?.contains(msg) == true)
    }
}

// MARK: - WhisperManager Unit Tests (no model load)

@available(macOS 14, iOS 17, *)
final class WhisperManagerStateTests: XCTestCase {

    func testManagerIsNotAvailableBeforeLoad() async {
        let manager = WhisperManager()
        let available = await manager.isAvailable
        XCTAssertFalse(available)
    }
}

// MARK: - WhisperModels Cache Path Tests (no model load)

@available(macOS 14, iOS 17, *)
final class WhisperModelsCacheTests: XCTestCase {

    func testDefaultCacheDirectoryContainsCorrectSubpath() {
        let dir = WhisperModels.defaultCacheDirectory()
        XCTAssertTrue(
            dir.path.hasSuffix("openai_whisper-large-v3-v20240930_turbo"),
            "Default cache dir should be the real 4-layer turbo, got: \(dir.path)"
        )
    }

    func testDefaultCacheDirectoryIsSubdirectoryOfModelsRoot() {
        let dir = WhisperModels.defaultCacheDirectory()
        let root = dir.deletingLastPathComponent().deletingLastPathComponent()
        // root should be the models root (whisperkit-coreml is one level up from the model dir)
        XCTAssertTrue(dir.path.hasPrefix(root.path))
    }

    func testModelsExistReturnsFalseForNonexistentPath() {
        let fake = URL(fileURLWithPath: "/tmp/does_not_exist_\(UUID().uuidString)")
        XCTAssertFalse(WhisperModels.modelsExist(at: fake))
    }

    func testModelsExistAtDownloadedPath() throws {
        // Model location is resolved via FLUIDAUDIO_MODELS_DIR (set when running swift test).
        let dir = WhisperModels.defaultCacheDirectory()
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip(
                "Whisper models not found at \(dir.path). "
                    + "Set FLUIDAUDIO_MODELS_DIR and run: fluidaudiocli download --dataset whisper-models"
            )
        }
        XCTAssertTrue(WhisperModels.modelsExist(at: dir),
            "modelsExist should return true for fully downloaded model dir")
    }

    func testModelRepoNameIsRealTurbo() {
        XCTAssertEqual(
            Repo.whisperLargeV3Turbo.rawValue,
            "argmaxinc/whisperkit-coreml/openai_whisper-large-v3-v20240930_turbo"
        )
        XCTAssertEqual(Repo.whisperLargeV3Turbo.remotePath, "argmaxinc/whisperkit-coreml")
        XCTAssertEqual(Repo.whisperLargeV3Turbo.subPath, "openai_whisper-large-v3-v20240930_turbo")
    }
}

// MARK: - Whisper Integration Tests (require downloaded models)

/// Integration tests that load real CoreML models and run inference on actual audio.
/// These are skipped automatically if the model directory is not present.
@available(macOS 14, iOS 17, *)
final class WhisperManagerIntegrationTests: XCTestCase {

    private var modelDir: URL!

    override func setUpWithError() throws {
        // Use WhisperModels.defaultCacheDirectory() which respects FLUIDAUDIO_MODELS_DIR.
        // Run integration tests with:
        //   FLUIDAUDIO_MODELS_DIR=/path/to/FluidAudio/Models swift test --filter WhisperManagerIntegrationTests
        let dir = WhisperModels.defaultCacheDirectory()
        guard WhisperModels.modelsExist(at: dir) else {
            throw XCTSkip(
                "Whisper models not present at \(dir.path). "
                + "Set FLUIDAUDIO_MODELS_DIR and run: fluidaudiocli download --dataset whisper-models"
            )
        }
        modelDir = dir
    }

    private func loadManager() async throws -> WhisperManager {
        let manager = WhisperManager()
        try await manager.loadModels(from: modelDir)
        let available = await manager.isAvailable
        XCTAssertTrue(available)
        return manager
    }

    private func makeAudioSamples(
        durationSeconds: Double = 3.0,
        frequency: Float = 440.0,
        amplitude: Float = 0.3
    ) -> [Float] {
        let sampleCount = Int(durationSeconds * Double(WhisperConfig.sampleRate))
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(WhisperConfig.sampleRate)
            return amplitude * sin(2 * .pi * frequency * t)
        }
    }

    private func projectRoot() -> URL {
        // Navigate from the process's working directory (swift test runs from package root)
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    // MARK: - Load tests

    func testLoadModels() async throws {
        let manager = WhisperManager()
        let before = await manager.isAvailable
        XCTAssertFalse(before)
        try await manager.loadModels(from: modelDir)
        let after = await manager.isAvailable
        XCTAssertTrue(after)
    }

    func testLoadModelsFromNonexistentPathThrows() async {
        let manager = WhisperManager()
        let bad = URL(fileURLWithPath: "/tmp/nonexistent_\(UUID().uuidString)")
        do {
            try await manager.loadModels(from: bad)
            XCTFail("Expected error loading from nonexistent path")
        } catch {
            // Expected
            let available = await manager.isAvailable
            XCTAssertFalse(available)
        }
    }

    // MARK: - Transcription tests (short audio)

    func testTranscribeReturnsString() async throws {
        let manager = try await loadManager()
        let audio = makeAudioSamples(durationSeconds: 3.0)
        let result = try await manager.transcribe(audioSamples: audio)
        // Tone audio may return empty or noise tokens — just assert no crash
        XCTAssertNotNil(result)
    }

    func testTranscribeVeryShortAudioDoesNotCrash() async throws {
        let manager = try await loadManager()
        // 0.5 s — shorter than typical speech but should not crash
        let audio = makeAudioSamples(durationSeconds: 0.5)
        let result = try await manager.transcribe(audioSamples: audio)
        XCTAssertNotNil(result)
    }

    func testTranscribeSilenceReturnsEmptyOrMinimal() async throws {
        let manager = try await loadManager()
        let silence = [Float](repeating: 0.0, count: WhisperConfig.sampleRate * 3)
        let text = try await manager.transcribe(audioSamples: silence, language: "en")
        // Silence should return empty or very short output (possibly "[Music]" etc.)
        XCTAssertLessThan(text.trimmingCharacters(in: .whitespacesAndNewlines).count, 50,
            "Silence should produce minimal output, got: \"\(text)\"")
    }

    func testTranscribeEnglishAudio() async throws {
        let manager = try await loadManager()
        let audioPath = projectRoot().appendingPathComponent("Tests/weanxinviec.mp3").path
        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("Test audio not found at \(audioPath)")
        }
        let samples = try AudioConverter().resampleAudioFile(path: audioPath)
        let text = try await manager.transcribe(audioSamples: samples, language: "en")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "Transcription of real audio should not be empty")
        print("[WhisperTest] weanxinviec.mp3 → \"\(text.prefix(120))\"")
    }

    func testTranscribeWithVietnameseLanguage() async throws {
        let manager = try await loadManager()
        let audioPath = projectRoot().appendingPathComponent("Tests/weanxinviec.mp3").path
        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("Test audio not found at \(audioPath)")
        }
        let samples = try AudioConverter().resampleAudioFile(path: audioPath)
        let text = try await manager.transcribe(audioSamples: samples, language: "vi")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "Vietnamese transcription should not be empty")
        print("[WhisperTest] weanxinviec.mp3 (vi) → \"\(text.prefix(120))\"")
    }

    func testTranscribeConsistency() async throws {
        // Two calls on the same real audio should return the same text.
        // Silence is intentionally avoided here: silence produces near-zero encoder output
        // which causes ANE non-determinism (different FP16 rounding paths each run).
        // Real speech has strong encoder activations that yield deterministic greedy output.
        let audioPath = projectRoot().appendingPathComponent("Tests/weanxinviec.mp3").path
        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("weanxinviec.mp3 not found — needed for determinism check")
        }
        let manager = try await loadManager()
        let samples = try AudioConverter().resampleAudioFile(path: audioPath)
        let text1 = try await manager.transcribe(audioSamples: samples, language: "vi")
        let text2 = try await manager.transcribe(audioSamples: samples, language: "vi")
        XCTAssertEqual(text1, text2, "Transcription must be deterministic for the same input")
    }

    // MARK: - Long audio (> 30 s)

    func testTranscribeLongAudioTriggersMultiWindow() async throws {
        // 35s of audio forces the multi-window code path (>windowSamples)
        let manager = try await loadManager()
        let sampleCount = Int(35.0 * Double(WhisperConfig.sampleRate))
        XCTAssertGreaterThan(sampleCount, WhisperConfig.windowSamples)
        let audio = (0..<sampleCount).map { i -> Float in
            0.2 * sin(2 * .pi * 220 * Float(i) / Float(WhisperConfig.sampleRate))
        }
        let text = try await manager.transcribe(audioSamples: audio, language: "en")
        XCTAssertNotNil(text)
    }

    func testTranscribeLongRealAudio() async throws {
        let audioPath = projectRoot().appendingPathComponent("Tests/bomman.mp3").path
        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("bomman.mp3 not found")
        }
        let manager = try await loadManager()
        let samples = try AudioConverter().resampleAudioFile(path: audioPath)
        let durationSeconds = Double(samples.count) / Double(WhisperConfig.sampleRate)
        print("[WhisperTest] bomman.mp3 duration: \(String(format: "%.1f", durationSeconds))s")
        // Long audio should not crash and return non-empty text
        let start = CFAbsoluteTimeGetCurrent()
        let text = try await manager.transcribe(audioSamples: samples, language: "en")
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let rtfx = durationSeconds / elapsed
        print("[WhisperTest] bomman.mp3 → RTFx \(String(format: "%.1f", rtfx))x, text prefix: \"\(text.prefix(80))\"")
        XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    // MARK: - RTFx / performance

    func testTranscribeRTFxIsReasonable() async throws {
        let manager = try await loadManager()
        // 10 seconds of audio → expect > 1× RTFx on M-series
        let audio = makeAudioSamples(durationSeconds: 10.0, frequency: 300.0)
        let audioSeconds = Double(audio.count) / Double(WhisperConfig.sampleRate)
        let start = CFAbsoluteTimeGetCurrent()
        _ = try await manager.transcribe(audioSamples: audio, language: "en")
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let rtfx = audioSeconds / elapsed
        print("[WhisperTest] 10s tone → RTFx: \(String(format: "%.1f", rtfx))x")
        XCTAssertGreaterThan(rtfx, 1.0, "Expected at least 1× real-time factor, got \(rtfx)×")
    }
}

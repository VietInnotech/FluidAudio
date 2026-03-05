import AVFoundation
import XCTest

@testable import FluidAudio

@available(macOS 14, iOS 17, *)
final class EraXWhisperIntegrationTests: XCTestCase {
    let testAudioSmall = "/Users/vit/offasr/FluidAudio/Tests/weanxinviec.mp3"

    private func loadManagerForVariant(_ variant: WhisperModelVariant) async throws -> WhisperManager {
        let manager = WhisperManager()
        let dir = WhisperModels.defaultCacheDirectory(variant: variant)
        guard WhisperModels.modelsExist(at: dir) else {
            throw XCTSkip("Models (\(variant.rawValue)) not found at \(dir.path)")
        }
        try await manager.loadModels(from: dir)
        return manager
    }

    func testEraXModelLoads() async throws {
        let manager = try await loadManagerForVariant(.eraXWowTurbo)
        let available = await manager.isAvailable
        XCTAssertTrue(available, "EraX model should be available after loading")
    }

    func testEraXTranscriptionSmallFile() async throws {
        let manager = try await loadManagerForVariant(.eraXWowTurbo)
        let samples = try AudioConverter().resampleAudioFile(path: testAudioSmall)
        let text = try await manager.transcribe(audioSamples: samples, language: "vi")
        XCTAssertFalse(text.isEmpty, "EraX transcription should not be empty")
        print("\n✓ EraX: \(text.prefix(100))")
    }

    func testStandardVsEraXComparison() async throws {
        let audioConverter = AudioConverter()
        let samples = try audioConverter.resampleAudioFile(path: testAudioSmall)

        print("\n" + String(repeating: "=", count: 60))
        print("COMPARATIVE TEST: Standard Whisper vs EraX-WoW-Turbo")
        print(String(repeating: "=", count: 60))

        // Standard Model
        let stdManager = try await loadManagerForVariant(.standard)
        let stdStart = Date()
        let stdText = try await stdManager.transcribe(audioSamples: samples, language: "vi")
        let stdTime = Date().timeIntervalSince(stdStart)
        print("\n📊 Standard Whisper:")
        print("  Time: \(String(format: "%.2f", stdTime))s")
        print("  Text: \(stdText.prefix(80))")

        // EraX Model
        let eraManager = try await loadManagerForVariant(.eraXWowTurbo)
        let eraStart = Date()
        let eraText = try await eraManager.transcribe(audioSamples: samples, language: "vi")
        let eraTime = Date().timeIntervalSince(eraStart)
        print("\n📊 EraX-WoW-Turbo:")
        print("  Time: \(String(format: "%.2f", eraTime))s")
        print("  Text: \(eraText.prefix(80))")

        XCTAssertFalse(stdText.isEmpty, "Standard should produce output")
        XCTAssertFalse(eraText.isEmpty, "EraX should produce output")
    }
}

import Foundation
import XCTest

@testable import FluidAudio

/// Quick API test for both standard and EraX Whisper models
@available(macOS 14, iOS 17, *)
final class WhisperAPITest: XCTestCase {

    private func loadManagerAndTranscribe(
        variant: WhisperModelVariant,
        language: String
    ) async throws -> String {
        let manager = WhisperManager()
        
        // Load models
        let dir = WhisperModels.defaultCacheDirectory(variant: variant)
        guard WhisperModels.modelsExist(at: dir) else {
            throw XCTSkip("Whisper (\(variant.rawValue)) models not at \(dir.path)")
        }
        try await manager.loadModels(from: dir)
        let available = await manager.isAvailable
        XCTAssertTrue(available)
        
        // Load test audio
        let audioPath = projectRoot()
            .appendingPathComponent("Tests/weanxinviec.mp3").path
        guard FileManager.default.fileExists(atPath: audioPath) else {
            throw XCTSkip("Test audio not found")
        }
        let samples = try AudioConverter().resampleAudioFile(path: audioPath)
        
        // Transcribe
        let text = try await manager.transcribe(audioSamples: samples, language: language)
        return text
    }

    func testStandardWhisperAPI() async throws {
        let text = try await loadManagerAndTranscribe(variant: .standard, language: "vi")
        XCTAssertFalse(text.isEmpty, "Standard Whisper should produce output")
        print("\n✓ Standard Whisper API: \(text.prefix(100))")
    }

    func testEraXWhisperAPI() async throws {
        let text = try await loadManagerAndTranscribe(variant: .eraXWowTurbo, language: "vi")
        XCTAssertFalse(text.isEmpty, "EraX should produce output")
        print("\n✓ EraX Whisper API: \(text.prefix(100))")
    }

    func testBothVariantsProduceOutput() async throws {
        let standard = try await loadManagerAndTranscribe(variant: .standard, language: "vi")
        let erax = try await loadManagerAndTranscribe(variant: .eraXWowTurbo, language: "vi")
        
        XCTAssertFalse(standard.isEmpty, "Standard should not be empty")
        XCTAssertFalse(erax.isEmpty, "EraX should not be empty")
        
        print("\n Standard: \(standard.prefix(80))")
        print("EraX:     \(erax.prefix(80))")
    }

    private func projectRoot() -> URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }
}

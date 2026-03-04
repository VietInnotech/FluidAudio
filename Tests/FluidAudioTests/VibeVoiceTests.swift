import Foundation
import XCTest

@testable import FluidAudio

// MARK: - VibeVoice-ASR Unit Tests (no models required)

/// Tests for VibeVoice-ASR configuration, output parser, error types, and model paths.
/// None of these tests require model files on disk.
@available(macOS 15, iOS 18, *)
final class VibeVoiceTests: XCTestCase {

    // MARK: - Config Tests

    func testConfigAudioConstants() {
        XCTAssertEqual(VibeVoiceAsrConfig.sampleRate, 24000)
        XCTAssertEqual(VibeVoiceAsrConfig.compressionRatio, 3200)
        XCTAssertEqual(VibeVoiceAsrConfig.maxAudioSeconds, 3600.0, accuracy: 0.01)
    }

    func testConfigTokenizerConstants() {
        XCTAssertEqual(VibeVoiceAsrConfig.acousticVaeDim, 64)
        XCTAssertEqual(VibeVoiceAsrConfig.semanticVaeDim, 128)
        XCTAssertEqual(VibeVoiceAsrConfig.acousticDownsampleRatios, [8, 5, 5, 4, 2, 2])
        // Product of ratios must equal compression ratio
        let product = VibeVoiceAsrConfig.acousticDownsampleRatios.reduce(1, *)
        XCTAssertEqual(product, VibeVoiceAsrConfig.compressionRatio)
    }

    func testConfigDecoderConstants() {
        XCTAssertEqual(VibeVoiceAsrConfig.hiddenSize, 3584)
        XCTAssertEqual(VibeVoiceAsrConfig.intermediateSize, 18944)
        XCTAssertEqual(VibeVoiceAsrConfig.numDecoderLayers, 28)
        XCTAssertEqual(VibeVoiceAsrConfig.numAttentionHeads, 28)
        XCTAssertEqual(VibeVoiceAsrConfig.numKVHeads, 4)
        XCTAssertEqual(VibeVoiceAsrConfig.headDim, 128)
        XCTAssertEqual(VibeVoiceAsrConfig.vocabSize, 152_064)
        XCTAssertEqual(VibeVoiceAsrConfig.ropeTheta, 1_000_000, accuracy: 0.01)
    }

    func testConfigHeadDimConsistency() {
        // headDim should equal hiddenSize / numAttentionHeads
        let expected = VibeVoiceAsrConfig.hiddenSize / VibeVoiceAsrConfig.numAttentionHeads
        XCTAssertEqual(VibeVoiceAsrConfig.headDim, expected)
    }

    func testConfigGQARatio() {
        // GQA ratio: 28 heads / 4 KV heads = 7
        let gqaRatio = VibeVoiceAsrConfig.numAttentionHeads / VibeVoiceAsrConfig.numKVHeads
        XCTAssertEqual(gqaRatio, 7)
    }

    func testConfigSpecialTokensAreDistinct() {
        let specialTokens = [
            VibeVoiceAsrConfig.speechStartTokenId,
            VibeVoiceAsrConfig.speechEndTokenId,
            VibeVoiceAsrConfig.speechPadTokenId,
            VibeVoiceAsrConfig.eotTokenId,
            VibeVoiceAsrConfig.imStartTokenId,
            VibeVoiceAsrConfig.imEndTokenId,
        ]
        XCTAssertEqual(Set(specialTokens).count, specialTokens.count, "Special tokens must be unique")
    }

    func testConfigSpecialTokensWithinVocab() {
        let specialTokens = [
            VibeVoiceAsrConfig.speechStartTokenId,
            VibeVoiceAsrConfig.speechEndTokenId,
            VibeVoiceAsrConfig.speechPadTokenId,
            VibeVoiceAsrConfig.eotTokenId,
            VibeVoiceAsrConfig.imStartTokenId,
            VibeVoiceAsrConfig.imEndTokenId,
        ]
        for token in specialTokens {
            XCTAssertLessThan(token, VibeVoiceAsrConfig.vocabSize, "Token \(token) exceeds vocab size")
        }
    }

    func testConfigEosTokenIds() {
        // Both eot and im_end should be EOS
        XCTAssertTrue(VibeVoiceAsrConfig.eosTokenIds.contains(VibeVoiceAsrConfig.eotTokenId))
        XCTAssertTrue(VibeVoiceAsrConfig.eosTokenIds.contains(VibeVoiceAsrConfig.imEndTokenId))
        XCTAssertEqual(VibeVoiceAsrConfig.eosTokenIds.count, 2)
    }

    func testConfigSystemPrompt() {
        XCTAssertFalse(VibeVoiceAsrConfig.systemPrompt.isEmpty)
        XCTAssertTrue(VibeVoiceAsrConfig.systemPrompt.contains("JSON"))
    }

    func testConfigOutputKeys() {
        XCTAssertEqual(VibeVoiceAsrConfig.outputKeys.count, 4)
        XCTAssertTrue(VibeVoiceAsrConfig.outputKeys.contains("Start time"))
        XCTAssertTrue(VibeVoiceAsrConfig.outputKeys.contains("End time"))
        XCTAssertTrue(VibeVoiceAsrConfig.outputKeys.contains("Speaker ID"))
        XCTAssertTrue(VibeVoiceAsrConfig.outputKeys.contains("Content"))
    }

    func testConfigGenerationDefaults() {
        XCTAssertGreaterThan(VibeVoiceAsrConfig.maxCacheSeqLen, 0)
        XCTAssertGreaterThan(VibeVoiceAsrConfig.defaultMaxNewTokens, 0)
        XCTAssertGreaterThanOrEqual(VibeVoiceAsrConfig.defaultMaxNewTokens, 1024)
    }

    // MARK: - Language Tests

    func testLanguageEnumAllCases() {
        XCTAssertGreaterThan(VibeVoiceAsrConfig.Language.allCases.count, 10)
    }

    func testLanguageFromISOCode() {
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "en"), .english)
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "zh"), .chinese)
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "fr"), .french)
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "ja"), .japanese)
    }

    func testLanguageFromEnglishName() {
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "English"), .english)
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "chinese"), .chinese)
        XCTAssertEqual(VibeVoiceAsrConfig.Language(from: "French"), .french)
    }

    func testLanguageFromInvalidReturnsNil() {
        XCTAssertNil(VibeVoiceAsrConfig.Language(from: "unknown_lang"))
        XCTAssertNil(VibeVoiceAsrConfig.Language(from: ""))
        XCTAssertNil(VibeVoiceAsrConfig.Language(from: "xx"))
    }

    func testLanguageEnglishNames() {
        XCTAssertEqual(VibeVoiceAsrConfig.Language.english.englishName, "English")
        XCTAssertEqual(VibeVoiceAsrConfig.Language.chinese.englishName, "Chinese")
        XCTAssertEqual(VibeVoiceAsrConfig.Language.japanese.englishName, "Japanese")
    }

    // MARK: - Output Parser Tests

    func testParseValidJSONArray() {
        let json = """
            [
              {"Start time": "0.00s", "End time": "2.50s", "Speaker ID": "Speaker 1", "Content": "Hello, how are you?"},
              {"Start time": "3.00s", "End time": "5.20s", "Speaker ID": "Speaker 2", "Content": "I'm doing well, thanks."}
            ]
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.segments.count, 2)

        XCTAssertEqual(result.segments[0].startTime, "0.00s")
        XCTAssertEqual(result.segments[0].endTime, "2.50s")
        XCTAssertEqual(result.segments[0].speakerId, "Speaker 1")
        XCTAssertEqual(result.segments[0].content, "Hello, how are you?")

        XCTAssertEqual(result.segments[1].startTime, "3.00s")
        XCTAssertEqual(result.segments[1].endTime, "5.20s")
        XCTAssertEqual(result.segments[1].speakerId, "Speaker 2")
        XCTAssertEqual(result.segments[1].content, "I'm doing well, thanks.")
    }

    func testParseSingleObject() {
        let json = """
            {"Start time": "0.00s", "End time": "10.00s", "Speaker ID": "Speaker 1", "Content": "Single utterance."}
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].content, "Single utterance.")
    }

    func testParseMarkdownCodeBlock() {
        let text = """
            ```json
            [
              {"Start time": "0.00s", "End time": "1.50s", "Speaker ID": "Speaker 1", "Content": "Markdown test."}
            ]
            ```
            """
        let result = VibeVoiceOutputParser.parse(text)
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].content, "Markdown test.")
    }

    func testParsePlainCodeBlock() {
        let text = """
            ```
            [
              {"Start time": "5.00s", "End time": "8.00s", "Speaker ID": "Speaker 2", "Content": "Plain block."}
            ]
            ```
            """
        let result = VibeVoiceOutputParser.parse(text)
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].content, "Plain block.")
    }

    func testParseTruncatedJSON() {
        // Simulates model generation that gets cut off mid-JSON
        let truncated = """
            [
              {"Start time": "0.00s", "End time": "2.50s", "Speaker ID": "Speaker 1", "Content": "Complete"},
              {"Start time": "3.00s", "End time": "5.00s", "Speaker ID": "Speaker 2", "Content": "Also complete"}
            """
        let result = VibeVoiceOutputParser.parse(truncated)
        // Should repair missing ] and parse successfully
        XCTAssertEqual(result.segments.count, 2)
    }

    func testParseTruncatedJSONWithTrailingComma() {
        let truncated = """
            [
              {"Start time": "0.00s", "End time": "2.50s", "Speaker ID": "Speaker 1", "Content": "First"},
            """
        let result = VibeVoiceOutputParser.parse(truncated)
        XCTAssertGreaterThanOrEqual(result.segments.count, 1)
    }

    func testParseEmptyText() {
        let result = VibeVoiceOutputParser.parse("")
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertEqual(result.rawText, "")
    }

    func testParseGarbageText() {
        let result = VibeVoiceOutputParser.parse("This is not JSON at all.")
        XCTAssertTrue(result.segments.isEmpty)
    }

    func testParseTextWithLeadingGarbage() {
        let text = """
            Some preamble text before the JSON
            [{"Start time": "1.00s", "End time": "3.00s", "Speaker ID": "Speaker 1", "Content": "After garbage."}]
            """
        let result = VibeVoiceOutputParser.parse(text)
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].content, "After garbage.")
    }

    // MARK: - Transcription Result Tests

    func testTranscriptionResultSpeakerCount() {
        let json = """
            [
              {"Start time": "0.00s", "End time": "2.00s", "Speaker ID": "Speaker 1", "Content": "A"},
              {"Start time": "2.00s", "End time": "4.00s", "Speaker ID": "Speaker 2", "Content": "B"},
              {"Start time": "4.00s", "End time": "6.00s", "Speaker ID": "Speaker 1", "Content": "C"},
              {"Start time": "6.00s", "End time": "8.00s", "Speaker ID": "Speaker 3", "Content": "D"}
            ]
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.speakerCount, 3)
    }

    func testTranscriptionResultTotalDuration() throws {
        let json = """
            [
              {"Start time": "0.00s", "End time": "5.00s", "Speaker ID": "Speaker 1", "Content": "A"},
              {"Start time": "5.00s", "End time": "12.50s", "Speaker ID": "Speaker 2", "Content": "B"}
            ]
            """
        let result = VibeVoiceOutputParser.parse(json)
        let duration = try XCTUnwrap(result.totalDuration)
        XCTAssertEqual(duration, 12.50, accuracy: 0.01)
    }

    func testTranscriptionResultPlainText() {
        let json = """
            [
              {"Start time": "0.00s", "End time": "2.00s", "Speaker ID": "Speaker 1", "Content": "Hello world."},
              {"Start time": "2.00s", "End time": "4.00s", "Speaker ID": "Speaker 2", "Content": "Goodbye world."}
            ]
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.plainText, "Hello world. Goodbye world.")
    }

    func testTranscriptionResultRawTextPreserved() {
        let rawText = "some raw output"
        let result = VibeVoiceOutputParser.parse(rawText)
        XCTAssertEqual(result.rawText, rawText)
    }

    // MARK: - Segment Time Parsing Tests

    func testSegmentTimeParsingWithSuffix() throws {
        let segment = VibeVoiceTranscriptionSegment(
            startTime: "1.50s",
            endTime: "3.75s",
            speakerId: "Speaker 1",
            content: "test"
        )
        let start = try XCTUnwrap(segment.startTimeSeconds)
        let end = try XCTUnwrap(segment.endTimeSeconds)
        XCTAssertEqual(start, 1.50, accuracy: 0.001)
        XCTAssertEqual(end, 3.75, accuracy: 0.001)
    }

    func testSegmentTimeParsingWithoutSuffix() throws {
        let segment = VibeVoiceTranscriptionSegment(
            startTime: "2.00",
            endTime: "5.00",
            speakerId: "Speaker 1",
            content: "test"
        )
        let start = try XCTUnwrap(segment.startTimeSeconds)
        let end = try XCTUnwrap(segment.endTimeSeconds)
        XCTAssertEqual(start, 2.00, accuracy: 0.001)
        XCTAssertEqual(end, 5.00, accuracy: 0.001)
    }

    func testSegmentEquatable() {
        let a = VibeVoiceTranscriptionSegment(
            startTime: "0.00s", endTime: "1.00s", speakerId: "Speaker 1", content: "Hello"
        )
        let b = VibeVoiceTranscriptionSegment(
            startTime: "0.00s", endTime: "1.00s", speakerId: "Speaker 1", content: "Hello"
        )
        let c = VibeVoiceTranscriptionSegment(
            startTime: "0.00s", endTime: "1.00s", speakerId: "Speaker 2", content: "Hello"
        )
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    func testSegmentCodableRoundTrip() throws {
        let segment = VibeVoiceTranscriptionSegment(
            startTime: "0.50s", endTime: "2.50s", speakerId: "Speaker 1", content: "Roundtrip test."
        )
        let encoder = JSONEncoder()
        let data = try encoder.encode(segment)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VibeVoiceTranscriptionSegment.self, from: data)
        XCTAssertEqual(segment, decoded)
    }

    func testSegmentCodingKeys() throws {
        // Verify the JSON keys match the VibeVoice output format
        let json = """
            {"Start time": "0.00s", "End time": "1.00s", "Speaker ID": "Speaker 1", "Content": "Keys test."}
            """
        let data = json.data(using: .utf8)!
        let segment = try JSONDecoder().decode(VibeVoiceTranscriptionSegment.self, from: data)
        XCTAssertEqual(segment.startTime, "0.00s")
        XCTAssertEqual(segment.speakerId, "Speaker 1")
        XCTAssertEqual(segment.content, "Keys test.")
    }

    // MARK: - Model Names Tests

    func testModelNamesVibeVoice() {
        XCTAssertEqual(ModelNames.VibeVoice.acousticEncoderFile, "vibevoice_acoustic_encoder.mlmodelc")
        XCTAssertEqual(ModelNames.VibeVoice.semanticEncoderFile, "vibevoice_semantic_encoder.mlmodelc")
        XCTAssertEqual(ModelNames.VibeVoice.decoderStatefulFile, "vibevoice_decoder_stateful.mlmodelc")
        XCTAssertEqual(ModelNames.VibeVoice.embeddingsFile, "vibevoice_embeddings.bin")
    }

    func testModelNamesRequiredModels() {
        let required = ModelNames.VibeVoice.requiredModels
        XCTAssertEqual(required.count, 4)
        XCTAssertTrue(required.contains("vibevoice_acoustic_encoder.mlmodelc"))
        XCTAssertTrue(required.contains("vibevoice_semantic_encoder.mlmodelc"))
        XCTAssertTrue(required.contains("vibevoice_decoder_stateful.mlmodelc"))
        XCTAssertTrue(required.contains("vibevoice_embeddings.bin"))
    }

    // MARK: - Repo Configuration Tests

    func testRepoVibeVoiceAsr() {
        let repo = Repo.vibevoiceAsr
        XCTAssertTrue(repo.rawValue.contains("vibevoice"))
        XCTAssertTrue(repo.name.contains("vibevoice"))
        XCTAssertTrue(repo.remotePath.contains("FluidInference"))
    }

    func testRepoVibeVoiceAsrInt4() {
        let repo = Repo.vibevoiceAsrInt4
        XCTAssertTrue(repo.rawValue.contains("int4"))
        XCTAssertTrue(repo.name.contains("int4"))
    }

    func testRepoVariantMapping() {
        XCTAssertEqual(VibeVoiceAsrVariant.f32.repo, .vibevoiceAsr)
        XCTAssertEqual(VibeVoiceAsrVariant.int4.repo, .vibevoiceAsrInt4)
    }

    func testVariantAllCases() {
        XCTAssertEqual(VibeVoiceAsrVariant.allCases.count, 2)
        XCTAssertTrue(VibeVoiceAsrVariant.allCases.contains(.f32))
        XCTAssertTrue(VibeVoiceAsrVariant.allCases.contains(.int4))
    }

    // MARK: - Error Description Tests

    func testErrorDescriptions() {
        let errors: [VibeVoiceAsrError] = [
            .modelLoadFailed("test"),
            .encoderFailed("test"),
            .decoderFailed("test"),
            .generationFailed("test"),
            .audioTooShort,
            .audioTooLong(7200.0),
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription, "Error \(error) should have description")
            XCTAssertFalse(error.errorDescription!.isEmpty, "Error \(error) description shouldn't be empty")
        }
    }

    func testErrorAudioTooLongContainsDuration() {
        let error = VibeVoiceAsrError.audioTooLong(7200.0)
        XCTAssertTrue(error.errorDescription!.contains("7200"))
    }

    // MARK: - Manager State Tests

    func testManagerInitialState() async {
        let manager = VibeVoiceAsrManager()
        let available = await manager.isAvailable
        XCTAssertFalse(available, "Manager should not be available before loading models")
    }

    func testManagerTranscribeWithoutModels() async {
        let manager = VibeVoiceAsrManager()
        do {
            let _ = try await manager.transcribe(audioSamples: [0.1, 0.2, 0.3])
            XCTFail("Expected error when transcribing without models")
        } catch {
            // Expected — models not loaded
            XCTAssertTrue(error is VibeVoiceAsrError)
        }
    }

    func testManagerTranscribeEmptyAudio() async {
        let manager = VibeVoiceAsrManager()
        // Even before models are loaded, empty audio should fail
        do {
            let _ = try await manager.transcribe(audioSamples: [])
            XCTFail("Expected error for empty audio")
        } catch {
            XCTAssertTrue(error is VibeVoiceAsrError)
        }
    }

    // MARK: - Models Exist Check

    func testModelsExistReturnsFalseForEmptyDir() {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        XCTAssertFalse(VibeVoiceAsrModels.modelsExist(at: tmpDir))
    }

    func testModelsExistReturnsFalseForNonexistentDir() {
        let fakeDir = URL(fileURLWithPath: "/tmp/nonexistent_vibevoice_test_dir_\(UUID().uuidString)")
        XCTAssertFalse(VibeVoiceAsrModels.modelsExist(at: fakeDir))
    }

    // MARK: - Complex Parser Scenarios

    func testParseMultiSpeakerConversation() throws {
        let json = """
            [
              {"Start time": "0.00s", "End time": "3.50s", "Speaker ID": "Speaker 1", "Content": "Good morning everyone, let's start the meeting."},
              {"Start time": "3.80s", "End time": "6.20s", "Speaker ID": "Speaker 2", "Content": "Sure, I have some updates on the project."},
              {"Start time": "6.50s", "End time": "12.00s", "Speaker ID": "Speaker 2", "Content": "We've completed the backend migration and the tests are all passing now."},
              {"Start time": "12.30s", "End time": "15.10s", "Speaker ID": "Speaker 3", "Content": "Great to hear. What about the frontend changes?"},
              {"Start time": "15.40s", "End time": "20.00s", "Speaker ID": "Speaker 2", "Content": "Those are in progress, should be done by end of week."},
              {"Start time": "20.30s", "End time": "22.80s", "Speaker ID": "Speaker 1", "Content": "Perfect. Let's move on to the next topic."}
            ]
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.segments.count, 6)
        XCTAssertEqual(result.speakerCount, 3)
        let multiDuration = try XCTUnwrap(result.totalDuration)
        XCTAssertEqual(multiDuration, 22.80, accuracy: 0.01)
        XCTAssertFalse(result.plainText.isEmpty)
    }

    func testParseUnicodeContent() {
        let json = """
            [
              {"Start time": "0.00s", "End time": "2.00s", "Speaker ID": "Speaker 1", "Content": "你好世界"},
              {"Start time": "2.00s", "End time": "4.00s", "Speaker ID": "Speaker 2", "Content": "こんにちは"},
              {"Start time": "4.00s", "End time": "6.00s", "Speaker ID": "Speaker 1", "Content": "مرحبا بالعالم"}
            ]
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.segments.count, 3)
        XCTAssertEqual(result.segments[0].content, "你好世界")
        XCTAssertEqual(result.segments[1].content, "こんにちは")
        XCTAssertEqual(result.segments[2].content, "مرحبا بالعالم")
    }

    func testParseEscapedQuotesInContent() throws {
        let json = """
            [{"Start time": "0.00s", "End time": "2.00s", "Speaker ID": "Speaker 1", "Content": "He said \\"hello\\" to me."}]
            """
        let result = VibeVoiceOutputParser.parse(json)
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertTrue(result.segments[0].content.contains("hello"))
    }

    func testParseEmptyArray() {
        let result = VibeVoiceOutputParser.parse("[]")
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertEqual(result.speakerCount, 0)
        XCTAssertNil(result.totalDuration)
        XCTAssertEqual(result.plainText, "")
    }
}

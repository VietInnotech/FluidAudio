import XCTest

@testable import FluidAudio

final class VadAsrPipelineTests: XCTestCase {

    private actor Counter {
        private(set) var value: Int = 0

        func increment() {
            value += 1
        }
    }

    func testTranscribeReturnsEmptyWhenNoSpeechSegments() async throws {
        let pipeline = VadAsrPipeline(
            segmentSpeech: { _, _ in [] },
            transcribe: { _, _ in
                XCTFail("ASR should not run when there are no VAD segments")
                return ASRResult(text: "", confidence: 0, duration: 0, processingTime: 0)
            }
        )

        let samples = Array(repeating: Float(0), count: 16_000)
        let result = try await pipeline.transcribe(samples)

        XCTAssertEqual(result.text, "")
        XCTAssertEqual(result.segments.count, 0)
        XCTAssertEqual(result.vadSegments.count, 0)
        XCTAssertEqual(result.confidence, 0)
        XCTAssertEqual(result.speechDuration, 0)
    }

    func testTranscribeRemapsTokenTimingsAndJoinsText() async throws {
        let segmentA = VadSegment(startTime: 1.0, endTime: 2.2)
        let segmentB = VadSegment(startTime: 5.0, endTime: 6.5)

        let pipeline = VadAsrPipeline(
            segmentSpeech: { _, _ in [segmentA, segmentB] },
            transcribe: { samples, _ in
                if samples.count < 20_000 {
                    return ASRResult(
                        text: "hello",
                        confidence: 0.9,
                        duration: 1.2,
                        processingTime: 0.1,
                        tokenTimings: [
                            TokenTiming(token: "hello", tokenId: 10, startTime: 0.2, endTime: 0.8, confidence: 0.9)
                        ]
                    )
                }

                return ASRResult(
                    text: "world",
                    confidence: 0.6,
                    duration: 1.5,
                    processingTime: 0.2,
                    tokenTimings: [
                        TokenTiming(token: "world", tokenId: 20, startTime: 0.1, endTime: 0.7, confidence: 0.6)
                    ]
                )
            }
        )

        let samples = Array(repeating: Float(0.1), count: 16_000 * 8)
        let result = try await pipeline.transcribe(samples)

        XCTAssertEqual(result.segments.count, 2)
        XCTAssertEqual(result.text, "hello world")
        XCTAssertEqual(result.vadSegments.count, 2)

        let firstTiming = try XCTUnwrap(result.segments[0].tokenTimings.first)
        XCTAssertEqual(firstTiming.startTime, 1.2, accuracy: 0.0001)
        XCTAssertEqual(firstTiming.endTime, 1.8, accuracy: 0.0001)

        let secondTiming = try XCTUnwrap(result.segments[1].tokenTimings.first)
        XCTAssertEqual(secondTiming.startTime, 5.1, accuracy: 0.0001)
        XCTAssertEqual(secondTiming.endTime, 5.7, accuracy: 0.0001)

        XCTAssertGreaterThan(result.speechRatio, 0)
        XCTAssertGreaterThan(result.rtfx, 0)
    }

    func testTranscribeSkipsSubSecondSegments() async throws {
        let shortSegment = VadSegment(startTime: 0.0, endTime: 0.5)
        let validSegment = VadSegment(startTime: 2.0, endTime: 3.5)
        let counter = Counter()

        let pipeline = VadAsrPipeline(
            segmentSpeech: { _, _ in [shortSegment, validSegment] },
            transcribe: { _, _ in
                await counter.increment()
                return ASRResult(text: "speech", confidence: 0.8, duration: 1.5, processingTime: 0.1)
            }
        )

        let samples = Array(repeating: Float(0.05), count: 16_000 * 5)
        let result = try await pipeline.transcribe(samples)
        let transcribeCalls = await counter.value

        XCTAssertEqual(transcribeCalls, 1)
        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.text, "speech")
    }
}
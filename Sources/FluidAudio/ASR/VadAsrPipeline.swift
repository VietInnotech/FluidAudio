import Foundation
import OSLog

/// VAD-guided batch ASR pipeline.
///
/// Runs Silero VAD segmentation first, then transcribes only speech segments and remaps
/// token timings back to absolute positions in the original audio.
public final class VadAsrPipeline {

    private let logger = AppLogger(category: "VadAsrPipeline")
    private let segmentationConfig: VadSegmentationConfig

    private let segmentSpeechImpl: (_ samples: [Float], _ config: VadSegmentationConfig) async throws -> [VadSegment]
    private let transcribeImpl: (_ samples: [Float], _ source: AudioSource) async throws -> ASRResult

    public init(
        vadManager: VadManager,
        asrManager: AsrManager,
        segmentationConfig: VadSegmentationConfig = .asrOptimized
    ) {
        self.segmentationConfig = segmentationConfig
        self.segmentSpeechImpl = { samples, config in
            try await vadManager.segmentSpeech(samples, config: config)
        }
        self.transcribeImpl = { samples, source in
            try await asrManager.transcribe(samples, source: source)
        }
    }

    public convenience init(
        vadConfig: VadConfig = .default,
        asrConfig: ASRConfig = .default,
        asrModelVersion: AsrModelVersion = .v3,
        segmentationConfig: VadSegmentationConfig = .asrOptimized
    ) async throws {
        let vadManager = try await VadManager(config: vadConfig)
        let asrModels = try await AsrModels.downloadAndLoad(version: asrModelVersion)
        let asrManager = AsrManager(config: asrConfig)
        try await asrManager.initialize(models: asrModels)

        self.init(
            vadManager: vadManager,
            asrManager: asrManager,
            segmentationConfig: segmentationConfig
        )
    }

    internal init(
        segmentationConfig: VadSegmentationConfig = .asrOptimized,
        segmentSpeech: @escaping (_ samples: [Float], _ config: VadSegmentationConfig) async throws -> [VadSegment],
        transcribe: @escaping (_ samples: [Float], _ source: AudioSource) async throws -> ASRResult
    ) {
        self.segmentationConfig = segmentationConfig
        self.segmentSpeechImpl = segmentSpeech
        self.transcribeImpl = transcribe
    }

    public func transcribe(
        _ samples: [Float],
        source: AudioSource = .system
    ) async throws -> VadAsrResult {
        let pipelineStart = Date()

        let vadStart = Date()
        let vadSegments = try await segmentSpeechImpl(samples, segmentationConfig)
        let vadTime = Date().timeIntervalSince(vadStart)

        let audioDuration = Double(samples.count) / Double(VadManager.sampleRate)
        let speechDuration = vadSegments.reduce(0.0) { $0 + $1.duration }

        guard !vadSegments.isEmpty else {
            return VadAsrResult(
                text: "",
                segments: [],
                vadSegments: [],
                confidence: 0,
                processingTime: Date().timeIntervalSince(pipelineStart),
                vadTime: vadTime,
                asrTime: 0,
                audioDuration: audioDuration,
                speechDuration: 0
            )
        }

        let asrStart = Date()
        var transcribedSegments: [VadAsrSegment] = []
        transcribedSegments.reserveCapacity(vadSegments.count)

        var weightedConfidenceSum: Double = 0
        var weightedDurationSum: Double = 0

        for segment in vadSegments {
            let startSample = max(0, segment.startSample(sampleRate: VadManager.sampleRate))
            let endSample = min(samples.count, segment.endSample(sampleRate: VadManager.sampleRate))

            guard endSample > startSample else { continue }

            let segmentSamples = Array(samples[startSample..<endSample])
            guard segmentSamples.count >= 16_000 else { continue }

            let result = try await transcribeImpl(segmentSamples, source)
            let trimmedText = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedText.isEmpty else { continue }

            let adjustedTimings = result.tokenTimings?.map { timing in
                TokenTiming(
                    token: timing.token,
                    tokenId: timing.tokenId,
                    startTime: timing.startTime + segment.startTime,
                    endTime: timing.endTime + segment.startTime,
                    confidence: timing.confidence
                )
            }

            let weightedDuration = result.duration > 0 ? result.duration : segment.duration
            weightedConfidenceSum += Double(result.confidence) * weightedDuration
            weightedDurationSum += weightedDuration

            transcribedSegments.append(
                VadAsrSegment(
                    text: trimmedText,
                    startTime: segment.startTime,
                    endTime: segment.endTime,
                    confidence: result.confidence,
                    tokenTimings: adjustedTimings ?? []
                )
            )
        }

        let asrTime = Date().timeIntervalSince(asrStart)
        let combinedText = transcribedSegments.map(\.text).joined(separator: " ")
        let avgConfidence = weightedDurationSum > 0 ? Float(weightedConfidenceSum / weightedDurationSum) : 0
        let totalTime = Date().timeIntervalSince(pipelineStart)

        logger.info(
            "VAD-ASR done: total=\(String(format: "%.3f", totalTime))s, VAD=\(String(format: "%.3f", vadTime))s, ASR=\(String(format: "%.3f", asrTime))s, segments=\(transcribedSegments.count)"
        )

        return VadAsrResult(
            text: combinedText,
            segments: transcribedSegments,
            vadSegments: vadSegments,
            confidence: avgConfidence,
            processingTime: totalTime,
            vadTime: vadTime,
            asrTime: asrTime,
            audioDuration: audioDuration,
            speechDuration: speechDuration
        )
    }
}

extension AsrManager {
    /// Convenience wrapper to run VAD-guided ASR with an existing manager.
    public func transcribeWithVad(
        _ samples: [Float],
        vadManager: VadManager,
        source: AudioSource = .system,
        segmentationConfig: VadSegmentationConfig = .asrOptimized
    ) async throws -> VadAsrResult {
        let pipeline = VadAsrPipeline(
            vadManager: vadManager,
            asrManager: self,
            segmentationConfig: segmentationConfig
        )
        return try await pipeline.transcribe(samples, source: source)
    }
}

extension VadSegmentationConfig {
    /// Segmentation defaults tuned for ASR chunk boundaries.
    public static let asrOptimized = VadSegmentationConfig(
        minSpeechDuration: 0.25,
        minSilenceDuration: 0.5,
        maxSpeechDuration: 14.0,
        speechPadding: 0.15,
        silenceThresholdForSplit: 0.3,
        negativeThreshold: nil,
        negativeThresholdOffset: 0.15,
        minSilenceAtMaxSpeech: 0.098,
        useMaxPossibleSilenceAtMaxSpeech: true
    )

    /// Segmentation defaults tuned for Qwen3-ASR.
    ///
    /// Qwen3's KV-cache is limited to 512 tokens. At ~13 audio tokens per second,
    /// the practical prompt limit is ~25s of audio (accounting for system/chat tokens).
    /// We use 25s max speech duration to stay safely within that budget.
    public static let qwen3Optimized = VadSegmentationConfig(
        minSpeechDuration: 0.25,
        minSilenceDuration: 0.5,
        maxSpeechDuration: 25.0,
        speechPadding: 0.15,
        silenceThresholdForSplit: 0.3,
        negativeThreshold: nil,
        negativeThresholdOffset: 0.15,
        minSilenceAtMaxSpeech: 0.098,
        useMaxPossibleSilenceAtMaxSpeech: true
    )
}

public struct VadAsrResult: Sendable {
    public let text: String
    public let segments: [VadAsrSegment]
    public let vadSegments: [VadSegment]
    public let confidence: Float
    public let processingTime: TimeInterval
    public let vadTime: TimeInterval
    public let asrTime: TimeInterval
    public let audioDuration: TimeInterval
    public let speechDuration: TimeInterval

    public var speechRatio: Double {
        guard audioDuration > 0 else { return 0 }
        return speechDuration / audioDuration
    }

    public var rtfx: Float {
        guard processingTime > 0 else { return 0 }
        return Float(audioDuration / processingTime)
    }

    public init(
        text: String,
        segments: [VadAsrSegment],
        vadSegments: [VadSegment],
        confidence: Float,
        processingTime: TimeInterval,
        vadTime: TimeInterval,
        asrTime: TimeInterval,
        audioDuration: TimeInterval,
        speechDuration: TimeInterval
    ) {
        self.text = text
        self.segments = segments
        self.vadSegments = vadSegments
        self.confidence = confidence
        self.processingTime = processingTime
        self.vadTime = vadTime
        self.asrTime = asrTime
        self.audioDuration = audioDuration
        self.speechDuration = speechDuration
    }
}

public struct VadAsrSegment: Sendable {
    public let text: String
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Float
    public let tokenTimings: [TokenTiming]

    public var duration: TimeInterval {
        return endTime - startTime
    }

    public init(
        text: String,
        startTime: TimeInterval,
        endTime: TimeInterval,
        confidence: Float,
        tokenTimings: [TokenTiming]
    ) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
        self.tokenTimings = tokenTimings
    }
}
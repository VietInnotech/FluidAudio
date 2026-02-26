import AVFoundation
@preconcurrency import CoreML
import Foundation
import OSLog

/// Actor-based CTC ASR manager for standalone CTC transcription.
///
/// Uses the same CTC model pipeline as `CtcKeywordSpotter` (MelSpectrogram → AudioEncoder → logits)
/// but feeds the log-probabilities into a greedy decoder instead of constrained keyword spotting.
///
/// Supports the Vietnamese-finetuned Parakeet CTC 0.6B model as well as other CTC variants.
///
/// Usage:
/// ```swift
/// let manager = CtcAsrManager()
/// try await manager.loadModels(variant: .ctcVietnamese)
/// let result = try await manager.transcribe(audioSamples: samples)
/// print(result.text)
/// ```
public actor CtcAsrManager {

    private let logger = AppLogger(category: "CtcAsrManager")

    /// Loaded CTC models (nil until `loadModels` is called).
    private var models: CtcModels?

    /// Audio converter for file-based transcription.
    private let audioConverter = AudioConverter()

    /// Internal keyword spotter used for its CTC inference pipeline (computeLogProbs).
    private var inferenceEngine: CtcKeywordSpotter?

    /// Greedy decoder instance (created after models are loaded).
    private var decoder: CtcGreedyDecoder?

    /// The model variant currently loaded.
    private var loadedVariant: CtcModelVariant?

    public init() {}

    // MARK: - Model Loading

    /// Load CTC models for the specified variant.
    ///
    /// Downloads models from HuggingFace if not already cached locally.
    ///
    /// - Parameters:
    ///   - variant: Which CTC model variant to load (default: `.ctcVietnamese`).
    ///   - directory: Optional custom cache directory. Uses default cache if nil.
    public func loadModels(
        variant: CtcModelVariant = .ctcVietnamese,
        directory: URL? = nil
    ) async throws {
        logger.info("Loading CTC models for \(variant.displayName)...")

        let ctcModels = try await CtcModels.downloadAndLoad(to: directory, variant: variant)

        // The blank token is the last token in the vocabulary (vocab_size index)
        let blankId = ctcModels.vocabulary.count
        logger.info(
            "Models loaded: \(ctcModels.vocabulary.count) vocab tokens, blankId=\(blankId)"
        )

        self.models = ctcModels
        self.inferenceEngine = CtcKeywordSpotter(models: ctcModels, blankId: blankId)
        self.decoder = CtcGreedyDecoder(vocabulary: ctcModels.vocabulary, blankId: blankId)
        self.loadedVariant = variant
    }

    /// Load CTC models from a pre-downloaded directory (no download).
    ///
    /// - Parameters:
    ///   - directory: Directory containing the CTC CoreML model bundles and vocab.json.
    ///   - variant: Which CTC model variant to load.
    public func loadModels(
        from directory: URL,
        variant: CtcModelVariant = .ctcVietnamese
    ) async throws {
        logger.info("Loading CTC models directly from: \(directory.path)")

        let ctcModels = try await CtcModels.loadDirect(from: directory, variant: variant)

        let blankId = ctcModels.vocabulary.count
        logger.info(
            "Models loaded: \(ctcModels.vocabulary.count) vocab tokens, blankId=\(blankId)"
        )

        self.models = ctcModels
        self.inferenceEngine = CtcKeywordSpotter(models: ctcModels, blankId: blankId)
        self.decoder = CtcGreedyDecoder(vocabulary: ctcModels.vocabulary, blankId: blankId)
        self.loadedVariant = variant
    }

    // MARK: - Transcription

    /// Transcribe raw audio samples using CTC greedy decoding.
    ///
    /// - Parameter audioSamples: 16kHz mono Float32 audio samples.
    /// - Returns: `ASRResult` with transcribed text, confidence, timings, and performance metrics.
    public func transcribe(audioSamples: [Float]) async throws -> ASRResult {
        guard let engine = inferenceEngine, let decoder = decoder else {
            throw ASRError.notInitialized
        }

        guard audioSamples.count >= ASRConstants.sampleRate else {
            throw ASRError.invalidAudioData
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // Run CTC inference pipeline: audio → mel → encoder → log-probs
        let logProbResult = try await engine.computeLogProbs(for: audioSamples)

        guard !logProbResult.logProbs.isEmpty else {
            throw ASRError.processingFailed("CTC inference produced empty log-probabilities")
        }

        // Greedy decode: argmax → collapse → blank removal → text
        let decodingResult = decoder.decode(
            logProbs: logProbResult.logProbs,
            frameDuration: logProbResult.frameDuration
        )

        let processingTime = CFAbsoluteTimeGetCurrent() - startTime
        let audioDuration = Double(audioSamples.count) / Double(ASRConstants.sampleRate)

        logger.info(
            "Transcribed \(String(format: "%.1f", audioDuration))s audio in \(String(format: "%.2f", processingTime))s"
                + " (\(String(format: "%.1f", audioDuration / processingTime))x RTF)"
        )

        return ASRResult(
            text: decodingResult.text,
            confidence: decodingResult.confidence,
            duration: audioDuration,
            processingTime: processingTime,
            tokenTimings: decodingResult.tokenTimings
        )
    }

    /// Transcribe an audio file using CTC greedy decoding.
    ///
    /// The file is automatically converted to 16kHz mono Float32.
    ///
    /// - Parameter url: Path to the audio file (WAV, M4A, CAF, etc.).
    /// - Returns: `ASRResult` with transcribed text.
    public func transcribe(url: URL) async throws -> ASRResult {
        let samples: [Float]
        do {
            samples = try audioConverter.resampleAudioFile(url)
        } catch {
            throw ASRError.fileAccessFailed(url, error)
        }
        return try await transcribe(audioSamples: samples)
    }

    // MARK: - Status

    /// Whether models are currently loaded and ready for transcription.
    public var isReady: Bool {
        models != nil
    }

    /// The currently loaded model variant, if any.
    public var variant: CtcModelVariant? {
        loadedVariant
    }
}

@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "VibeVoiceAsrModels")

// MARK: - VibeVoice-ASR CoreML Model Container

/// Holds CoreML model components for VibeVoice-ASR inference.
///
/// Components:
/// - `acousticEncoder`: raw audio → acoustic VAE latent features
/// - `semanticEncoder`: raw audio → semantic embeddings
/// - `decoderStateful`: stateful Qwen2.5-7B decoder with fused lmHead (outputs logits)
/// - `embeddingWeights`: float16 matrix for Swift-side embedding lookup
///
/// The inference pipeline mirrors the Python VibeVoice-ASR processor:
/// 1. Audio (24kHz) → acoustic encoder → acoustic features
/// 2. Audio (24kHz) → semantic encoder → semantic features
/// 3. Merge features + prompt → Qwen2.5-7B decoder → structured text tokens
@available(macOS 15, iOS 18, *)
public struct VibeVoiceAsrModels: Sendable {
    public let acousticEncoder: MLModel
    public let semanticEncoder: MLModel
    public let decoderStateful: MLModel
    public let embeddingWeights: EmbeddingWeights
    public let vocabulary: [Int: String]

    /// Text decoder hidden size (3584 for Qwen2.5-7B).
    public var hiddenSize: Int { VibeVoiceAsrConfig.hiddenSize }

    /// Load VibeVoice-ASR models from a directory.
    ///
    /// Expected directory structure:
    /// ```
    /// vibevoice-asr-coreml/
    ///   vibevoice_acoustic_encoder.mlmodelc
    ///   vibevoice_semantic_encoder.mlmodelc
    ///   vibevoice_decoder_stateful.mlmodelc
    ///   vibevoice_embeddings.bin (float16 embedding weights)
    ///   vocab.json
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> VibeVoiceAsrModels {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        logger.info("Loading VibeVoice-ASR models from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load CoreML models in parallel
        async let acousticModel = loadModel(
            named: "vibevoice_acoustic_encoder",
            from: directory,
            computeUnits: .cpuOnly  // Encoder ops map best to CPU
        )
        async let semanticModel = loadModel(
            named: "vibevoice_semantic_encoder",
            from: directory,
            computeUnits: .cpuOnly
        )
        async let decoderModel = loadModel(
            named: "vibevoice_decoder_stateful",
            from: directory,
            computeUnits: computeUnits
        )
        async let embeddings = loadEmbeddings(from: directory)
        async let vocab = loadVocabulary(from: directory)

        let acoustic = try await acousticModel
        let semantic = try await semanticModel
        let decoder = try await decoderModel
        let embeddingWeights = try await embeddings
        let vocabulary = try await vocab

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info(
            "VibeVoice-ASR models loaded in \(String(format: "%.2f", elapsed))s (vocab: \(vocabulary.count) tokens, embeddings: \(embeddingWeights.vocabSize)x\(embeddingWeights.hiddenSize))"
        )

        return VibeVoiceAsrModels(
            acousticEncoder: acoustic,
            semanticEncoder: semantic,
            decoderStateful: decoder,
            embeddingWeights: embeddingWeights,
            vocabulary: vocabulary
        )
    }

    /// Create optimized prediction options.
    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        return MLPredictionOptions()
    }

    // MARK: - Download

    /// Download VibeVoice-ASR models from HuggingFace.
    ///
    /// Downloads CoreML models from FluidInference/vibevoice-asr-coreml.
    /// - Parameters:
    ///   - directory: Target directory (default: cache directory).
    ///   - variant: Model variant to download.
    ///   - force: Force re-download even if models exist.
    /// - Returns: Path to the directory containing the downloaded models.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        variant: VibeVoiceAsrVariant = .f32,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory(for: variant)
        let modelsRoot = targetDir.deletingLastPathComponent().deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("VibeVoice-ASR models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            try? FileManager.default.removeItem(at: targetDir)
        }

        logger.info("Downloading VibeVoice-ASR CoreML models from HuggingFace...")
        try await DownloadUtils.downloadRepo(variant.repo, to: modelsRoot)

        logger.info("Successfully downloaded VibeVoice-ASR models")
        return targetDir
    }

    /// Default cache directory for VibeVoice-ASR models.
    public static func defaultCacheDirectory(for variant: VibeVoiceAsrVariant = .f32) -> URL {
        ModelCachePaths.modelsRootDirectory()
            .appendingPathComponent(variant.repo.folderName, isDirectory: true)
    }

    /// Check if all required VibeVoice-ASR model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        return ModelNames.VibeVoice.requiredModels.allSatisfy { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }
    }

    // MARK: - Private Helpers

    private static func loadModel(
        named name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc")
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
        logger.info("Loaded \(name)")
        return model
    }

    private static func loadEmbeddings(from directory: URL) async throws -> EmbeddingWeights {
        let embeddingsURL = directory.appendingPathComponent(ModelNames.VibeVoice.embeddingsFile)
        return try EmbeddingWeights.load(from: embeddingsURL)
    }

    private static func loadVocabulary(from directory: URL) async throws -> [Int: String] {
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let data = try Data(contentsOf: vocabURL)
        let rawVocab = try JSONDecoder().decode([String: Int].self, from: data)

        var vocabulary: [Int: String] = [:]
        vocabulary.reserveCapacity(rawVocab.count)
        for (token, id) in rawVocab {
            vocabulary[id] = token
        }

        logger.info("Loaded vocabulary: \(vocabulary.count) tokens")
        return vocabulary
    }
}

// MARK: - Model Variant

/// VibeVoice-ASR model variant (precision).
public enum VibeVoiceAsrVariant: String, CaseIterable, Sendable {
    /// Full precision (FP16 weights). Best quality, ~14 GB for decoder.
    case f32

    /// INT4 quantized weights. ~3.5 GB for decoder. Recommended for on-device use.
    case int4

    /// Corresponding HuggingFace model repository.
    public var repo: Repo {
        switch self {
        case .f32: return .vibevoiceAsr
        case .int4: return .vibevoiceAsrInt4
        }
    }
}

// MARK: - Errors

public enum VibeVoiceAsrError: Error, LocalizedError {
    case modelLoadFailed(String)
    case encoderFailed(String)
    case decoderFailed(String)
    case generationFailed(String)
    case audioTooShort
    case audioTooLong(Double)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return "VibeVoice model load failed: \(msg)"
        case .encoderFailed(let msg): return "VibeVoice encoder failed: \(msg)"
        case .decoderFailed(let msg): return "VibeVoice decoder failed: \(msg)"
        case .generationFailed(let msg): return "VibeVoice generation failed: \(msg)"
        case .audioTooShort: return "Audio is too short to process"
        case .audioTooLong(let duration):
            let maxSecs: Double
            if #available(macOS 15, iOS 18, *) {
                maxSecs = VibeVoiceAsrConfig.maxAudioSeconds
            } else {
                maxSecs = 3600.0
            }
            return "Audio duration \(String(format: "%.1f", duration))s exceeds maximum of \(Int(maxSecs))s"
        }
    }
}

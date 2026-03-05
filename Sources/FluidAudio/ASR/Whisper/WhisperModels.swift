@preconcurrency import CoreML
import Foundation
import Hub
import OSLog
import Tokenizers

private let logger = Logger(subsystem: "FluidAudio", category: "WhisperModels")

// MARK: - Whisper Model Variant

/// Selects which Whisper CoreML model to download and load.
public enum WhisperModelVariant: String, CaseIterable, Sendable {
    /// OpenAI Whisper Large-v3 Turbo (standard, from argmaxinc/whisperkit-coreml).
    case standard
    /// EraX-WoW-Turbo V1.1 — fine-tuned whisper-large-v3-turbo for Vietnamese + 10 languages.
    case eraXWowTurbo
}

// MARK: - Whisper Models Container

/// Holds all CoreML models and tokenizer needed for Whisper inference.
/// Adapted from WhisperKit (MIT License, Copyright © 2024 Argmax, Inc.)
public struct WhisperModels: Sendable {
    /// Mel spectrogram extraction model
    public let melSpectrogram: MLModel
    /// Audio encoder model
    public let audioEncoder: MLModel
    /// Text decoder model
    public let textDecoder: MLModel
    /// Context prefill model (optional — accelerates prefill)
    public let contextPrefill: MLModel?
    /// Tokenizer for encoding/decoding text
    public let tokenizer: any Tokenizers.Tokenizer
    /// Maximum token context length for KV cache, read from the TextDecoder model's input description.
    /// Standard WhisperKit models use 224; EraX uses 448.
    public let maxTokenContext: Int

    // MARK: - Model Loading

    /// Load all Whisper models from a directory.
    ///
    /// Uses per-model compute units matching WhisperKit defaults:
    /// - MelSpectrogram: `.cpuAndGPU`
    /// - AudioEncoder: `.cpuAndNeuralEngine`
    /// - TextDecoder: `.cpuAndNeuralEngine`
    /// - TextDecoderContextPrefill: `.cpuOnly`
    ///
    /// Expected directory contents:
    /// - MelSpectrogram.mlmodelc
    /// - AudioEncoder.mlmodelc
    /// - TextDecoder.mlmodelc
    /// - TextDecoderContextPrefill.mlmodelc (optional)
    /// - tokenizer.json + tokenizer_config.json (for swift-transformers)
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> WhisperModels {
        let start = CFAbsoluteTimeGetCurrent()

        // Per-model compute unit assignment matching WhisperKit's ModelComputeOptions defaults.
        // Using `.all` for all models causes NaN in encoder output on Apple Neural Engine.
        let melUnits: MLComputeUnits = .cpuAndGPU
        let encoderUnits: MLComputeUnits = .cpuAndNeuralEngine
        let decoderUnits: MLComputeUnits = .cpuAndNeuralEngine
        let prefillUnits: MLComputeUnits = .cpuOnly

        // Load CoreML models in parallel
        async let melModel = loadModel(
            named: "MelSpectrogram",
            from: directory,
            computeUnits: melUnits
        )
        async let encoderModel = loadModel(
            named: "AudioEncoder",
            from: directory,
            computeUnits: encoderUnits
        )
        async let decoderModel = loadModel(
            named: "TextDecoder",
            from: directory,
            computeUnits: decoderUnits
        )
        async let prefillModel = loadModelOptional(
            named: "TextDecoderContextPrefill",
            from: directory,
            computeUnits: prefillUnits
        )

        let mel = try await melModel
        let encoder = try await encoderModel
        let decoder = try await decoderModel
        let prefill = await prefillModel

        // Load tokenizer from directory using swift-transformers
        let tokenizer = try await loadTokenizer(from: directory)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Whisper models loaded in \(String(format: "%.2f", elapsed))s")

        // Read maxTokenContext from the TextDecoder model's key_cache input shape.
        // This allows supporting models compiled with different KV cache sequence lengths
        // (e.g. standard=224, EraX=448) without hardcoded constants.
        let maxTokenContext: Int
        if let keyCacheDesc = decoder.modelDescription.inputDescriptionsByName["key_cache"],
            let constraint = keyCacheDesc.multiArrayConstraint
        {
            // key_cache shape is [1, kvCacheEmbedDim, 1, maxTokenContext]
            let shape = constraint.shape.map { $0.intValue }
            maxTokenContext = shape.count >= 4 ? shape[3] : WhisperConfig.maxTokenContext
            logger.info("TextDecoder KV cache sequence length: \(maxTokenContext)")
        } else {
            maxTokenContext = WhisperConfig.maxTokenContext
            logger.warning("Could not read KV cache shape from TextDecoder, using default \(maxTokenContext)")
        }

        return WhisperModels(
            melSpectrogram: mel,
            audioEncoder: encoder,
            textDecoder: decoder,
            contextPrefill: prefill,
            tokenizer: tokenizer,
            maxTokenContext: maxTokenContext
        )
    }

    /// Create optimized prediction options.
    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()
        return options
    }

    // MARK: - Download

    /// Download Whisper Large v3 Turbo models from HuggingFace.
    ///
    /// Downloads CoreML models from argmaxinc/whisperkit-coreml and tokenizer files
    /// from openai/whisper-large-v3-turbo (tokenizer is not bundled in the WhisperKit repo).
    /// - Returns: Path to the directory containing the downloaded models.
    @discardableResult
    public static func download(to directory: URL? = nil, force: Bool = false) async throws -> URL {
        try await download(variant: .standard, to: directory, force: force)
    }

    /// Download Whisper models for the specified variant.
    ///
    /// - Parameters:
    ///   - variant: Which model to download (`.standard` = argmaxinc WhisperKit, `.eraXWowTurbo` = EraX fine-tune).
    ///   - directory: Override the local cache directory. `nil` uses the default cache.
    ///   - force: Re-download even if models already exist locally.
    /// - Returns: Path to the directory containing the downloaded models.
    @discardableResult
    public static func download(
        variant: WhisperModelVariant,
        to directory: URL? = nil,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory(variant: variant)
        let modelsRoot = ModelCachePaths.modelsRootDirectory()

        if !force && modelsExist(at: targetDir) {
            logger.info("Whisper models (\(variant.rawValue)) already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            try? FileManager.default.removeItem(at: targetDir)
        }

        let repo: Repo = variant == .eraXWowTurbo ? .eraXWowTurbo : .whisperLargeV3Turbo
        logger.info("Downloading Whisper CoreML models (\(variant.rawValue)) from HuggingFace...")
        try await DownloadUtils.downloadRepo(repo, to: modelsRoot)

        // The WhisperKit repo does not bundle tokenizer files — download them separately.
        logger.info("Downloading Whisper tokenizer files from openai/whisper-large-v3-turbo...")
        try await downloadTokenizerFiles(to: targetDir)

        logger.info("Successfully downloaded Whisper models (\(variant.rawValue))")
        return targetDir
    }

    /// Default cache directory for the standard Whisper model.
    public static func defaultCacheDirectory() -> URL {
        defaultCacheDirectory(variant: .standard)
    }

    /// Default cache directory for a specific variant.
    public static func defaultCacheDirectory(variant: WhisperModelVariant) -> URL {
        let repo: Repo = variant == .eraXWowTurbo ? .eraXWowTurbo : .whisperLargeV3Turbo
        return ModelCachePaths.modelsRootDirectory()
            .appendingPathComponent(repo.folderName, isDirectory: true)
    }

    /// Check if all required Whisper model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let requiredFiles = Array(ModelNames.Whisper.requiredModels) + ["tokenizer.json"]
        return requiredFiles.allSatisfy { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }
    }

    /// Download tokenizer files from openai/whisper-large-v3-turbo.
    private static func downloadTokenizerFiles(to directory: URL) async throws {
        let tokenizerFiles = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ]
        let fm = FileManager.default
        for fileName in tokenizerFiles {
            let destPath = directory.appendingPathComponent(fileName)
            guard !fm.fileExists(atPath: destPath.path) else { continue }
            let fileURL = try ModelRegistry.resolveModel("openai/whisper-large-v3-turbo", fileName)
            let data = try await DownloadUtils.fetchHuggingFaceFile(
                from: fileURL,
                description: "Whisper tokenizer: \(fileName)"
            )
            try data.write(to: destPath)
            logger.info("Downloaded \(fileName)")
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

    private static func loadModelOptional(
        named name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) async -> MLModel? {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            logger.info("\(name) not found, skipping")
            return nil
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        do {
            let model = try await MLModel.load(contentsOf: modelURL, configuration: config)
            logger.info("Loaded \(name)")
            return model
        } catch {
            logger.warning("Failed to load optional \(name): \(error.localizedDescription)")
            return nil
        }
    }

    private static func loadTokenizer(from directory: URL) async throws -> any Tokenizers.Tokenizer {
        let tokenizerJSON = directory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerJSON.path) else {
            throw WhisperError.modelLoadFailed("tokenizer.json not found in \(directory.path)")
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        logger.info("Tokenizer loaded")
        return tokenizer
    }
}

// MARK: - MLFeatureProvider Wrappers

/// Input provider for MelSpectrogram model.
final class WhisperMelInput: MLFeatureProvider {
    let audio: MLMultiArray

    var featureNames: Set<String> { ["audio"] }

    init(audio: MLMultiArray) {
        self.audio = audio
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "audio" else { return nil }
        return MLFeatureValue(multiArray: audio)
    }
}

/// Input provider for AudioEncoder model.
final class WhisperEncoderInput: MLFeatureProvider {
    let melspectrogramFeatures: MLMultiArray

    var featureNames: Set<String> { ["melspectrogram_features"] }

    init(melspectrogramFeatures: MLMultiArray) {
        self.melspectrogramFeatures = melspectrogramFeatures
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == "melspectrogram_features" else { return nil }
        return MLFeatureValue(multiArray: melspectrogramFeatures)
    }
}

/// Input provider for TextDecoder model.
final class WhisperDecoderInput: MLFeatureProvider {
    let inputIds: MLMultiArray
    let cacheLength: MLMultiArray
    let keyCache: MLMultiArray
    let valueCache: MLMultiArray
    let kvCacheUpdateMask: MLMultiArray
    let encoderOutputEmbeds: MLMultiArray
    let decoderKeyPaddingMask: MLMultiArray

    var featureNames: Set<String> {
        [
            "input_ids", "cache_length", "key_cache", "value_cache",
            "kv_cache_update_mask", "encoder_output_embeds", "decoder_key_padding_mask",
        ]
    }

    init(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        kvCacheUpdateMask: MLMultiArray,
        encoderOutputEmbeds: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) {
        self.inputIds = inputIds
        self.cacheLength = cacheLength
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.kvCacheUpdateMask = kvCacheUpdateMask
        self.encoderOutputEmbeds = encoderOutputEmbeds
        self.decoderKeyPaddingMask = decoderKeyPaddingMask
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids": return MLFeatureValue(multiArray: inputIds)
        case "cache_length": return MLFeatureValue(multiArray: cacheLength)
        case "key_cache": return MLFeatureValue(multiArray: keyCache)
        case "value_cache": return MLFeatureValue(multiArray: valueCache)
        case "kv_cache_update_mask": return MLFeatureValue(multiArray: kvCacheUpdateMask)
        case "encoder_output_embeds": return MLFeatureValue(multiArray: encoderOutputEmbeds)
        case "decoder_key_padding_mask": return MLFeatureValue(multiArray: decoderKeyPaddingMask)
        default: return nil
        }
    }
}

/// Input provider for TextDecoderContextPrefill model.
final class WhisperPrefillInput: MLFeatureProvider {
    let task: MLMultiArray
    let language: MLMultiArray

    var featureNames: Set<String> { ["task", "language"] }

    init(task: MLMultiArray, language: MLMultiArray) {
        self.task = task
        self.language = language
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "task": return MLFeatureValue(multiArray: task)
        case "language": return MLFeatureValue(multiArray: language)
        default: return nil
        }
    }
}

// MARK: - Error Types

/// Errors that can occur during Whisper inference.
public enum WhisperError: Error, LocalizedError {
    case modelLoadFailed(String)
    case melExtractionFailed(String)
    case encodingFailed(String)
    case decodingFailed(String)
    case tokenizerError(String)
    case audioTooShort

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .melExtractionFailed(let msg): return "Mel extraction failed: \(msg)"
        case .encodingFailed(let msg): return "Encoding failed: \(msg)"
        case .decodingFailed(let msg): return "Decoding failed: \(msg)"
        case .tokenizerError(let msg): return "Tokenizer error: \(msg)"
        case .audioTooShort: return "Audio is too short to process"
        }
    }
}

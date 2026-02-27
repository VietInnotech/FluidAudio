@preconcurrency import CoreML
import Foundation
import Hub
import OSLog
import Tokenizers

private let logger = Logger(subsystem: "FluidAudio", category: "WhisperModels")

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

    // MARK: - Model Loading

    /// Load all Whisper models from a directory.
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

        // Load CoreML models in parallel
        async let melModel = loadModel(
            named: "MelSpectrogram",
            from: directory,
            computeUnits: computeUnits
        )
        async let encoderModel = loadModel(
            named: "AudioEncoder",
            from: directory,
            computeUnits: computeUnits
        )
        async let decoderModel = loadModel(
            named: "TextDecoder",
            from: directory,
            computeUnits: computeUnits
        )
        async let prefillModel = loadModelOptional(
            named: "TextDecoderContextPrefill",
            from: directory,
            computeUnits: computeUnits
        )

        let mel = try await melModel
        let encoder = try await encoderModel
        let decoder = try await decoderModel
        let prefill = await prefillModel

        // Load tokenizer from directory using swift-transformers
        let tokenizer = try await loadTokenizer(from: directory)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Whisper models loaded in \(String(format: "%.2f", elapsed))s")

        return WhisperModels(
            melSpectrogram: mel,
            audioEncoder: encoder,
            textDecoder: decoder,
            contextPrefill: prefill,
            tokenizer: tokenizer
        )
    }

    /// Create optimized prediction options.
    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()
        return options
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

@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "WhisperManager")

// MARK: - Whisper Manager

/// Public actor for Whisper Large v3 Turbo transcription.
///
/// Provides a clean API for loading models and transcribing audio, following
/// the same patterns as `Qwen3AsrManager`.
///
/// Pipeline:
/// 1. Audio → MelSpectrogram model → mel features [1, 128, 1, 3000]
/// 2. Mel features → AudioEncoder → encoder output [1, 1280, 1, 1500]
/// 3. (Optional) Prefill model → initial KV cache for SOT/language/task tokens
/// 4. Autoregressive decode loop with KV cache → token sequence
/// 5. Tokenizer decode → text
///
/// Adapted from WhisperKit (MIT License, Copyright © 2024 Argmax, Inc.)
@available(macOS 14, iOS 17, *)
public actor WhisperManager {
    private var models: WhisperModels?
    private lazy var predictionOptions: MLPredictionOptions = {
        WhisperModels.optimizedPredictionOptions()
    }()

    public init() {}

    /// Load all Whisper models from the specified directory.
    ///
    /// Expected contents: MelSpectrogram.mlmodelc, AudioEncoder.mlmodelc,
    /// TextDecoder.mlmodelc, TextDecoderContextPrefill.mlmodelc (optional),
    /// tokenizer.json, tokenizer_config.json
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await WhisperModels.load(from: directory, computeUnits: computeUnits)
        logger.info("Whisper models loaded successfully")
    }

    /// Whether all required models are loaded and ready.
    public var isAvailable: Bool {
        models != nil
    }

    /// Transcribe raw audio samples.
    ///
    /// - Parameters:
    ///   - audioSamples: 16kHz mono Float32 audio samples.
    ///   - language: Optional language code (e.g. "en", "fr"). nil for English default.
    ///   - options: Decoding options. nil uses defaults (greedy, English, no timestamps).
    /// - Returns: Transcribed text.
    public func transcribe(
        audioSamples: [Float],
        language: String? = "en",
        options: WhisperDecodingOptions? = nil
    ) async throws -> String {
        guard let models = models else {
            throw WhisperError.decodingFailed("Models not loaded. Call loadModels() first.")
        }

        let opts = options ?? WhisperDecodingOptions(language: language)
        let start = CFAbsoluteTimeGetCurrent()
        let audioLength = Float(audioSamples.count) / Float(WhisperConfig.sampleRate)

        // For audio longer than 30s, process in windows
        if audioSamples.count > WhisperConfig.windowSamples {
            let result = try await transcribeLongAudio(
                audioSamples: audioSamples,
                options: opts,
                models: models
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let rtfx = audioLength / Float(elapsed)
            logger.info(
                "Transcribed \(String(format: "%.1f", audioLength))s audio in \(String(format: "%.2f", elapsed))s (RTFx: \(String(format: "%.1f", rtfx))x)"
            )
            return result
        }

        // Single window transcription
        let result = try await transcribeWindow(
            audioSamples: audioSamples,
            options: opts,
            models: models
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let rtfx = audioLength / Float(elapsed)
        logger.info(
            "Transcribed \(String(format: "%.1f", audioLength))s audio in \(String(format: "%.2f", elapsed))s (RTFx: \(String(format: "%.1f", rtfx))x)"
        )

        return result.text
    }

    // MARK: - Single Window Transcription

    /// Transcribe a single ≤30s audio window.
    private func transcribeWindow(
        audioSamples: [Float],
        options: WhisperDecodingOptions,
        models: WhisperModels
    ) async throws -> WhisperDecodingResult {
        let opts = predictionOptions

        // Step 1: Extract mel spectrogram
        let melStart = CFAbsoluteTimeGetCurrent()
        let melFeatures = try await WhisperDecoder.extractMelSpectrogram(
            audio: audioSamples,
            model: models.melSpectrogram,
            options: opts
        )
        let melTime = CFAbsoluteTimeGetCurrent() - melStart
        logger.debug("Mel spectrogram: \(String(format: "%.3f", melTime))s")

        // Step 2: Encode audio
        let encodeStart = CFAbsoluteTimeGetCurrent()
        let encoderOutput = try await WhisperDecoder.encodeAudio(
            melFeatures: melFeatures,
            model: models.audioEncoder,
            options: opts
        )
        let encodeTime = CFAbsoluteTimeGetCurrent() - encodeStart
        logger.debug("Audio encoding: \(String(format: "%.3f", encodeTime))s")

        // Step 3: Prepare decoder state
        let state = try WhisperDecodingState(
            kvCacheEmbedDim: WhisperConfig.kvCacheEmbedDim,
            maxTokenContext: WhisperConfig.maxTokenContext
        )

        // Build prompt tokens
        let promptTokens = WhisperDecoder.buildPromptTokens(
            language: options.language,
            task: options.task,
            withoutTimestamps: options.withoutTimestamps
        )
        state.initialPrompt = promptTokens

        // Step 4: Prefill (optional)
        if options.usePrefillCache, let prefillModel = models.contextPrefill {
            let prefillStart = CFAbsoluteTimeGetCurrent()
            let languageTokenId = WhisperConfig.languageTokenId(for: options.language)
            let taskTokenId = options.task == .transcribe
                ? WhisperConfig.Tokens.transcribe : WhisperConfig.Tokens.translate

            let prefillCache = try await WhisperDecoder.prefillCache(
                model: prefillModel,
                languageTokenId: languageTokenId,
                taskTokenId: taskTokenId,
                options: opts
            )

            // Copy prefill cache into state
            let sotIndex = promptTokens.firstIndex(of: WhisperConfig.Tokens.startOfTranscript) ?? 0
            WhisperDecoder.updateKVCache(
                keyTensor: state.keyCache,
                keySlice: prefillCache.keyCache,
                valueTensor: state.valueCache,
                valueSlice: prefillCache.valueCache,
                insertAtIndex: sotIndex
            )

            let prefillSize = prefillCache.keyCache.shape[3].intValue
            state.cacheLength[0] = NSNumber(value: Int32(prefillSize))

            // Update masks for prefilled positions via direct pointer
            let totalPrefilled = prefillSize + 1  // +1 for initial mask
            let maskPtr = state.kvCacheUpdateMask.dataPointer.bindMemory(
                to: Float16.self, capacity: WhisperConfig.maxTokenContext)
            let padPtr = state.decoderKeyPaddingMask.dataPointer.bindMemory(
                to: Float16.self, capacity: WhisperConfig.maxTokenContext)
            for i in 0..<totalPrefilled {
                maskPtr[i] = 0
                padPtr[i] = 0
            }
            maskPtr[totalPrefilled - 1] = 1

            let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
            logger.debug("Prefill: \(String(format: "%.3f", prefillTime))s (cache size: \(prefillSize))")
        }

        // Step 5: Decode
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var result = try await WhisperDecoder.decodeText(
            encoderOutput: encoderOutput,
            decoderModel: models.textDecoder,
            state: state,
            decodingOptions: options,
            predictionOptions: opts
        )
        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        logger.debug("Decode: \(String(format: "%.3f", decodeTime))s (\(result.tokens.count) tokens)")

        // Step 6: Decode tokens to text using tokenizer
        let textTokens = options.withoutTimestamps
            ? result.tokens.filter { $0 < WhisperConfig.Tokens.specialTokenBegin }
            : result.tokens
        let text = models.tokenizer.decode(tokens: textTokens)
        result = WhisperDecodingResult(
            text: text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines),
            tokens: result.tokens,
            tokenLogProbs: result.tokenLogProbs,
            avgLogProb: result.avgLogProb,
            compressionRatio: result.compressionRatio,
            language: result.language
        )

        return result
    }

    // MARK: - Long Audio Transcription

    /// Transcribe audio longer than 30 seconds by windowing.
    private func transcribeLongAudio(
        audioSamples: [Float],
        options: WhisperDecodingOptions,
        models: WhisperModels
    ) async throws -> String {
        let windowSamples = WhisperConfig.windowSamples
        var allText: [String] = []
        var offset = 0

        while offset < audioSamples.count {
            let remaining = audioSamples.count - offset
            let chunkSize = min(windowSamples, remaining)

            // Skip very short trailing chunks
            if chunkSize < WhisperConfig.sampleRate {
                break
            }

            let chunk = Array(audioSamples[offset..<(offset + chunkSize)])
            let result = try await transcribeWindow(
                audioSamples: chunk,
                options: options,
                models: models
            )

            if !result.text.isEmpty {
                allText.append(result.text)
            }

            offset += windowSamples
        }

        return allText.joined(separator: " ")
    }
}

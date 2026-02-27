import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "WhisperDecoder")

// MARK: - Decoding State

/// Mutable state for a single decoding window.
/// Adapted from WhisperKit's DecodingInputs (MIT License, Copyright © 2024 Argmax, Inc.)
final class WhisperDecodingState {
    /// Initial prompt tokens (SOT, language, task, timestamps)
    var initialPrompt: [Int]
    /// Current token input (shape [1], Int32)
    let inputIds: MLMultiArray
    /// Cache length counter (shape [1], Int32)
    let cacheLength: MLMultiArray
    /// Key cache (shape [1, kvCacheEmbedDim, 1, maxTokenContext], Float16)
    let keyCache: MLMultiArray
    /// Value cache (shape [1, kvCacheEmbedDim, 1, maxTokenContext], Float16)
    let valueCache: MLMultiArray
    /// One-hot mask for cache write position (shape [1, maxTokenContext], Float16)
    let kvCacheUpdateMask: MLMultiArray
    /// Padding mask for decoder cross-attention (shape [1, maxTokenContext], Float16)
    let decoderKeyPaddingMask: MLMultiArray

    init(kvCacheEmbedDim: Int, maxTokenContext: Int) throws {
        let embedDimN = NSNumber(value: kvCacheEmbedDim)
        let maxSeqN = NSNumber(value: maxTokenContext)

        self.initialPrompt = []
        self.inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        self.cacheLength = try MLMultiArray(shape: [1], dataType: .int32)
        self.keyCache = try MLMultiArray(
            shape: [1, embedDimN, 1, maxSeqN], dataType: .float16)
        self.valueCache = try MLMultiArray(
            shape: [1, embedDimN, 1, maxSeqN], dataType: .float16)
        self.kvCacheUpdateMask = try MLMultiArray(
            shape: [1, maxSeqN], dataType: .float16)
        self.decoderKeyPaddingMask = try MLMultiArray(
            shape: [1, maxSeqN], dataType: .float16)

        // Initialize: all cache positions masked out, first position ready for write
        inputIds[0] = 0 as NSNumber
        cacheLength[0] = 0 as NSNumber

        // Initialize masks: kvCacheUpdateMask all 0 except [0]=1, padding mask all -10000 except [0]=0
        let maskPtr = kvCacheUpdateMask.dataPointer.bindMemory(to: Float16.self, capacity: maxTokenContext)
        let padPtr = decoderKeyPaddingMask.dataPointer.bindMemory(to: Float16.self, capacity: maxTokenContext)
        for i in 0..<maxTokenContext {
            maskPtr[i] = 0
            padPtr[i] = Float16(-10000)
        }
        maskPtr[0] = 1
        padPtr[0] = 0
    }

    /// Reset state for a new decoding window (reuse cache arrays).
    func reset(prefilledCacheSize: Int, maxTokenContext: Int) {
        cacheLength[0] = NSNumber(value: prefilledCacheSize)
        let maskPtr = kvCacheUpdateMask.dataPointer.bindMemory(to: Float16.self, capacity: maxTokenContext)
        let padPtr = decoderKeyPaddingMask.dataPointer.bindMemory(to: Float16.self, capacity: maxTokenContext)
        for i in 0..<maxTokenContext {
            if i <= prefilledCacheSize {
                padPtr[i] = 0
                if i > 0 { maskPtr[i - 1] = 0 }
                maskPtr[i] = 1
            } else {
                padPtr[i] = Float16(-10000)
                maskPtr[i] = 0
            }
        }
    }
}

// MARK: - Decoder Result

/// Result of decoding a single 30-second window.
public struct WhisperDecodingResult: Sendable {
    public let text: String
    public let tokens: [Int]
    public let tokenLogProbs: [Float]
    public let avgLogProb: Float
    public let compressionRatio: Float
    public let language: String
}

// MARK: - Whisper Decoder

/// Handles the full Whisper decode loop including KV cache management.
/// Adapted from WhisperKit's TextDecoder (MIT License, Copyright © 2024 Argmax, Inc.)
struct WhisperDecoder {

    // MARK: - Mel Spectrogram Extraction

    /// Extract mel spectrogram from raw audio using the CoreML MelSpectrogram model.
    static func extractMelSpectrogram(
        audio: [Float],
        model: MLModel,
        options: MLPredictionOptions
    ) async throws -> MLMultiArray {
        // Pad or trim audio to exactly 30 seconds (480,000 samples)
        var paddedAudio = audio
        if paddedAudio.count < WhisperConfig.windowSamples {
            paddedAudio.append(contentsOf: [Float](repeating: 0, count: WhisperConfig.windowSamples - paddedAudio.count))
        } else if paddedAudio.count > WhisperConfig.windowSamples {
            paddedAudio = Array(paddedAudio.prefix(WhisperConfig.windowSamples))
        }

        let audioArray = try MLMultiArray(shape: [NSNumber(value: WhisperConfig.windowSamples)], dataType: .float16)
        let ptr = audioArray.dataPointer.bindMemory(to: Float16.self, capacity: WhisperConfig.windowSamples)
        for i in 0..<WhisperConfig.windowSamples {
            ptr[i] = Float16(paddedAudio[i])
        }

        let input = WhisperMelInput(audio: audioArray)
        let output = try await model.prediction(from: input, options: options)

        guard let melFeatures = output.featureValue(for: "melspectrogram_features")?.multiArrayValue else {
            throw WhisperError.melExtractionFailed("No melspectrogram_features in output")
        }

        return melFeatures
    }

    // MARK: - Audio Encoding

    /// Encode mel spectrogram features using the AudioEncoder model.
    static func encodeAudio(
        melFeatures: MLMultiArray,
        model: MLModel,
        options: MLPredictionOptions
    ) async throws -> MLMultiArray {
        let input = WhisperEncoderInput(melspectrogramFeatures: melFeatures)
        let output = try await model.prediction(from: input, options: options)

        guard let encoderOutput = output.featureValue(for: "encoder_output_embeds")?.multiArrayValue else {
            throw WhisperError.encodingFailed("No encoder_output_embeds in output")
        }

        return encoderOutput
    }

    // MARK: - Prefill

    /// Run context prefill to get initial KV cache values for SOT+language+task tokens.
    static func prefillCache(
        model: MLModel,
        languageTokenId: Int,
        taskTokenId: Int,
        options: MLPredictionOptions
    ) async throws -> (keyCache: MLMultiArray, valueCache: MLMultiArray) {
        // Remap task token: 0 = transcribe, 1 = translate
        let taskValue: Int32 = taskTokenId == WhisperConfig.Tokens.transcribe ? 0 : 1
        let taskArray = try MLMultiArray(shape: [1], dataType: .int32)
        taskArray[0] = NSNumber(value: taskValue)

        let langArray = try MLMultiArray(shape: [1], dataType: .int32)
        langArray[0] = NSNumber(value: Int32(languageTokenId))

        let input = WhisperPrefillInput(task: taskArray, language: langArray)
        let output = try await model.prediction(from: input, options: options)

        guard let keyCachePrefill = output.featureValue(for: "key_cache_prefill")?.multiArrayValue,
            let valueCachePrefill = output.featureValue(for: "value_cache_prefill")?.multiArrayValue
        else {
            throw WhisperError.decodingFailed("Prefill model did not return key/value cache")
        }

        return (keyCachePrefill, valueCachePrefill)
    }

    // MARK: - Build Prompt

    /// Build the initial prompt tokens for a window.
    static func buildPromptTokens(
        language: String?,
        task: WhisperTask,
        withoutTimestamps: Bool
    ) -> [Int] {
        var tokens: [Int] = [WhisperConfig.Tokens.startOfTranscript]

        // Language token (multilingual model)
        let langTokenId = WhisperConfig.languageTokenId(for: language)
        tokens.append(langTokenId)

        // Task token
        let taskTokenId = task == .transcribe ? WhisperConfig.Tokens.transcribe : WhisperConfig.Tokens.translate
        tokens.append(taskTokenId)

        // Timestamp control
        if withoutTimestamps {
            tokens.append(WhisperConfig.Tokens.noTimestamps)
        } else {
            tokens.append(WhisperConfig.Tokens.timeTokenBegin)
        }

        return tokens
    }

    // MARK: - Decode Text (Main Loop)

    /// Decode text from encoder output for a single 30-second window.
    static func decodeText(
        encoderOutput: MLMultiArray,
        decoderModel: MLModel,
        state: WhisperDecodingState,
        decodingOptions: WhisperDecodingOptions,
        predictionOptions: MLPredictionOptions
    ) async throws -> WhisperDecodingResult {
        let maxTokenContext = WhisperConfig.maxTokenContext

        let initialPrompt = state.initialPrompt
        let prefilledIndex = state.cacheLength[0].intValue
        let initialPromptIndex = initialPrompt.count

        var currentTokens: [Int] = initialPrompt
        var nextToken: Int = initialPrompt.last!
        var logProbs: [Float] = Array(repeating: 0, count: currentTokens.count)

        let loopCount = min(decodingOptions.sampleLength, maxTokenContext - 1)

        // Build suppress token set for fast lookup
        let suppressSet = Set(WhisperConfig.suppressTokens)

        for tokenIndex in prefilledIndex..<loopCount {
            let isPrefill = tokenIndex < initialPromptIndex - 1

            // Force prompt tokens during prefill phase
            if tokenIndex < initialPromptIndex {
                nextToken = currentTokens[tokenIndex]
            }

            // Set current token and cache position
            state.inputIds[0] = NSNumber(value: Int32(nextToken))
            state.cacheLength[0] = NSNumber(value: Int32(tokenIndex))

            // Run decoder prediction
            let decoderInput = WhisperDecoderInput(
                inputIds: state.inputIds,
                cacheLength: state.cacheLength,
                keyCache: state.keyCache,
                valueCache: state.valueCache,
                kvCacheUpdateMask: state.kvCacheUpdateMask,
                encoderOutputEmbeds: encoderOutput,
                decoderKeyPaddingMask: state.decoderKeyPaddingMask
            )

            let output = try await decoderModel.prediction(from: decoderInput, options: predictionOptions)

            guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
                throw WhisperError.decodingFailed("No logits in decoder output")
            }

            // Apply logits filtering
            let filteredLogits = applyLogitsFilters(
                logits: logits,
                tokens: currentTokens,
                tokenIndex: tokenIndex,
                prefilledIndex: prefilledIndex,
                initialPromptIndex: initialPromptIndex,
                suppressSet: suppressSet,
                suppressBlank: decodingOptions.suppressBlank,
                withoutTimestamps: decodingOptions.withoutTimestamps
            )

            // Greedy decode (temperature = 0)
            let (sampledToken, tokenLogProb) = greedySample(logits: filteredLogits)
            nextToken = sampledToken

            // Check for first-token log-prob threshold
            let isFirstToken = tokenIndex == prefilledIndex
            var isFirstTokenLogProbTooLow = false
            if isFirstToken,
                let threshold = decodingOptions.firstTokenLogProbThreshold,
                tokenLogProb < threshold
            {
                isFirstTokenLogProbTooLow = true
            }

            // Check completion
            let isComplete =
                nextToken == WhisperConfig.Tokens.endOfText
                || currentTokens.count >= maxTokenContext - 1
                || isFirstTokenLogProbTooLow

            if isComplete {
                break
            }

            // Append token (skip during prefill)
            if !isPrefill {
                currentTokens.append(nextToken)
                logProbs.append(tokenLogProb)
            }

            // Update KV cache
            guard let newKeyCache = output.featureValue(for: "key_cache_updates")?.multiArrayValue,
                let newValueCache = output.featureValue(for: "value_cache_updates")?.multiArrayValue
            else {
                throw WhisperError.decodingFailed("Missing KV cache updates")
            }

            updateKVCache(
                keyTensor: state.keyCache,
                keySlice: newKeyCache,
                valueTensor: state.valueCache,
                valueSlice: newValueCache,
                insertAtIndex: tokenIndex
            )

            // Update masks for next position via direct pointer access
            let padPtr = state.decoderKeyPaddingMask.dataPointer.bindMemory(
                to: Float16.self, capacity: WhisperConfig.maxTokenContext)
            padPtr[tokenIndex + 1] = 0
            let maskPtr = state.kvCacheUpdateMask.dataPointer.bindMemory(
                to: Float16.self, capacity: WhisperConfig.maxTokenContext)
            maskPtr[tokenIndex] = 0
            maskPtr[tokenIndex + 1] = 1
        }

        // Finalize: ensure EOT is present
        if currentTokens.last != WhisperConfig.Tokens.endOfText {
            currentTokens.append(WhisperConfig.Tokens.endOfText)
            logProbs.append(0)
        }

        // Extract text tokens (filter out special tokens)
        let textTokens = currentTokens.filter { $0 < WhisperConfig.Tokens.specialTokenBegin }
        let avgLogProb = logProbs.isEmpty ? 0 : logProbs.reduce(0, +) / Float(logProbs.count)
        let compressionRatio = Self.compressionRatio(of: textTokens)

        // Decode tokens to text
        let text = decodingOptions.withoutTimestamps
            ? decodeTextTokens(textTokens, state: state)
            : decodeAllTokens(currentTokens, state: state)

        return WhisperDecodingResult(
            text: text,
            tokens: currentTokens,
            tokenLogProbs: logProbs,
            avgLogProb: avgLogProb,
            compressionRatio: compressionRatio,
            language: decodingOptions.language ?? WhisperConfig.defaultLanguageCode
        )
    }

    // MARK: - Token Decoding Helpers

    private static func decodeTextTokens(_ tokens: [Int], state: WhisperDecodingState) -> String {
        // Use a simple integer-to-string decode
        // swift-transformers tokenizer handles the BPE decoding
        return ""  // Placeholder — WhisperManager will handle tokenizer decode
    }

    private static func decodeAllTokens(_ tokens: [Int], state: WhisperDecodingState) -> String {
        return ""  // Placeholder — WhisperManager will handle tokenizer decode
    }

    // MARK: - KV Cache Update

    /// Copy KV cache slices into the full cache tensor at the given position.
    /// Uses direct memory access for performance.
    static func updateKVCache(
        keyTensor: MLMultiArray,
        keySlice: MLMultiArray,
        valueTensor: MLMultiArray,
        valueSlice: MLMultiArray,
        insertAtIndex index: Int
    ) {
        let tensorShape = keyTensor.shape.map { $0.intValue }
        let sliceShape = keySlice.shape.map { $0.intValue }
        let sliceStrides = keySlice.strides.map { $0.intValue }
        let bytesPerSample = MemoryLayout<Float16>.size

        keyTensor.withUnsafeMutableBytes { keyTensorPointer, keyTargetStrides in
            keySlice.withUnsafeBytes { keySlicePointer in
                valueTensor.withUnsafeMutableBytes { valueTensorPointer, valueTargetStrides in
                    valueSlice.withUnsafeBytes { valueSlicePointer in
                        DispatchQueue.concurrentPerform(iterations: tensorShape[1]) { j in
                            for k in 0..<sliceShape[3] {
                                let keyDestIndex = j * keyTargetStrides[1] + (index + k) * keyTargetStrides[3]
                                let keyDest = keyTensorPointer.baseAddress! + keyDestIndex * bytesPerSample
                                let keySliceIndex = j * sliceStrides[1] + k * sliceStrides[3]
                                let keySrc = keySlicePointer.baseAddress! + keySliceIndex * bytesPerSample
                                memcpy(keyDest, keySrc, bytesPerSample)

                                let valDestIndex = j * valueTargetStrides[1] + (index + k) * valueTargetStrides[3]
                                let valDest = valueTensorPointer.baseAddress! + valDestIndex * bytesPerSample
                                let valSliceIndex = j * sliceStrides[1] + k * sliceStrides[3]
                                let valSrc = valueSlicePointer.baseAddress! + valSliceIndex * bytesPerSample
                                memcpy(valDest, valSrc, bytesPerSample)
                            }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Logits Filtering

    /// Apply all logits filters: suppress tokens, suppress blank, timestamp rules.
    static func applyLogitsFilters(
        logits: MLMultiArray,
        tokens: [Int],
        tokenIndex: Int,
        prefilledIndex: Int,
        initialPromptIndex: Int,
        suppressSet: Set<Int>,
        suppressBlank: Bool,
        withoutTimestamps: Bool
    ) -> MLMultiArray {
        // The logits shape is [1, 1, vocabSize] for this model
        let vocabSize = logits.shape.last!.intValue

        // 1. Suppress tokens
        logits.withUnsafeMutableBytes { ptr, strides in
            let base = ptr.baseAddress!.bindMemory(to: Float16.self, capacity: vocabSize)
            for token in suppressSet {
                if token < vocabSize {
                    base[token] = Float16(-Float.infinity)
                }
            }

            // 2. Suppress blank at beginning
            if suppressBlank && tokens.count == prefilledIndex {
                // Suppress whitespace and EOT at very first decoded token
                base[WhisperConfig.Tokens.whitespace] = Float16(-Float.infinity)
                base[WhisperConfig.Tokens.endOfText] = Float16(-Float.infinity)
            }

            // 3. Timestamp rules (only when timestamps are enabled)
            if !withoutTimestamps {
                applyTimestampRules(base: base, vocabSize: vocabSize, tokens: tokens, initialPromptIndex: initialPromptIndex)
            }
        }

        return logits
    }

    /// Apply Whisper's timestamp constraint rules.
    private static func applyTimestampRules(
        base: UnsafeMutablePointer<Float16>,
        vocabSize: Int,
        tokens: [Int],
        initialPromptIndex: Int
    ) {
        let timeBegin = WhisperConfig.Tokens.timeTokenBegin
        let eot = WhisperConfig.Tokens.endOfText

        // Suppress noTimestamps
        if WhisperConfig.Tokens.noTimestamps < vocabSize {
            base[WhisperConfig.Tokens.noTimestamps] = Float16(-Float.infinity)
        }

        // Token count after initial prompt
        let sampledCount = tokens.count - initialPromptIndex

        guard sampledCount >= 1 else { return }

        let lastToken = tokens.last!
        let lastIsTimestamp = lastToken >= timeBegin

        if sampledCount >= 2 {
            let penultimate = tokens[tokens.count - 2]
            let penultimateIsTimestamp = penultimate >= timeBegin

            if lastIsTimestamp {
                if penultimateIsTimestamp {
                    // Two consecutive timestamps → force text tokens (no more timestamps)
                    for i in timeBegin..<vocabSize {
                        base[i] = Float16(-Float.infinity)
                    }
                } else {
                    // Single timestamp → must be followed by timestamp or EOT (close the pair)
                    for i in 0..<eot {
                        base[i] = Float16(-Float.infinity)
                    }
                }
            }
        }

        // Timestamps can't decrease
        if lastIsTimestamp && lastToken > timeBegin {
            for i in timeBegin..<lastToken {
                base[i] = Float16(-Float.infinity)
            }
        }

        // If sum of timestamp probs > max text prob, force timestamp
        var maxTextLogit: Float = -Float.infinity
        for i in 0..<eot {
            let v = Float(base[i])
            if v > maxTextLogit { maxTextLogit = v }
        }

        var timestampLogitSum: Float = 0
        for i in timeBegin..<vocabSize {
            timestampLogitSum += Float(base[i])
        }

        if timestampLogitSum > maxTextLogit {
            for i in 0..<eot {
                base[i] = Float16(-Float.infinity)
            }
        }
    }

    // MARK: - Greedy Sampling

    /// Greedy argmax sampling from logits. Returns (token, logProb).
    static func greedySample(logits: MLMultiArray) -> (Int, Float) {
        let vocabSize = logits.shape.last!.intValue
        var bestToken = 0
        var bestValue: Float = -Float.infinity

        logits.withUnsafeBytes { ptr in
            let base = ptr.baseAddress!.bindMemory(to: Float16.self, capacity: vocabSize)

            for i in 0..<vocabSize {
                let v = Float(base[i])
                if v > bestValue {
                    bestValue = v
                    bestToken = i
                }
            }
        }

        // Compute log softmax for the selected token
        // logProb = logit - log(sum(exp(logits)))
        var logSumExp: Float = 0
        logits.withUnsafeBytes { ptr in
            let base = ptr.baseAddress!.bindMemory(to: Float16.self, capacity: vocabSize)
            // Numerical stability: subtract max
            for i in 0..<vocabSize {
                let v = Float(base[i]) - bestValue
                logSumExp += exp(v)
            }
        }
        let logProb = bestValue - bestValue - log(logSumExp)  // = -log(sumExp)

        return (bestToken, logProb)
    }

    // MARK: - Compression Ratio

    /// Compute compression ratio of tokens (used for fallback detection).
    static func compressionRatio(of tokens: [Int]) -> Float {
        guard !tokens.isEmpty else { return 0 }
        let text = tokens.map { String($0) }.joined(separator: " ")
        let data = Data(text.utf8)
        // Simple approximation: ratio of original to "compressed" length
        // Use zlib-style estimation
        let uniqueTokens = Set(tokens)
        let ratio = Float(tokens.count) / Float(max(uniqueTokens.count, 1))
        return ratio
    }
}

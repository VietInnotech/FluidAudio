import Accelerate
@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "VibeVoiceAsrManager")

// MARK: - VibeVoice-ASR Manager

/// Manages VibeVoice-ASR CoreML inference for unified ASR + speaker diarization + timestamping.
///
/// VibeVoice-ASR is a unified speech-to-text model by Microsoft that handles up to 60 minutes
/// of audio in a single pass, generating structured transcriptions containing:
/// - **Who**: Speaker identification
/// - **When**: Timestamps
/// - **What**: Transcribed content
///
/// Pipeline:
/// 1. Audio (24kHz) → acoustic tokenizer encoder → acoustic features
/// 2. Audio (24kHz) → semantic tokenizer encoder → semantic features
/// 3. Build chat prompt with speech token placeholders
/// 4. Merge audio features into prompt → Qwen2.5-7B decoder → structured JSON
/// 5. Parse JSON output → typed `VibeVoiceTranscriptionSegment` structs
///
/// Usage:
/// ```swift
/// let manager = VibeVoiceAsrManager()
/// try await manager.loadModels(from: modelDirectory)
/// let result = try await manager.transcribe(audioSamples: samples)
/// for segment in result.segments {
///     print("[\(segment.startTime)-\(segment.endTime)] \(segment.speakerId): \(segment.content)")
/// }
/// ```
@available(macOS 15, iOS 18, *)
public actor VibeVoiceAsrManager {
    private var models: VibeVoiceAsrModels?
    /// Stored decoder state - held here so it can be explicitly nil-ed between
    /// transcription calls, ensuring the previous KV-cache (~224 MB) is freed
    /// before the next one is allocated.
    private var decoderState: MLState?
    private let audioConverter: AudioConverter = AudioConverter()

    public init() {}

    /// Load all VibeVoice-ASR models from the specified directory.
    ///
    /// Expected contents: vibevoice_acoustic_encoder.mlmodelc, vibevoice_semantic_encoder.mlmodelc,
    /// vibevoice_decoder_stateful.mlmodelc, vibevoice_embeddings.bin, vocab.json
    public func loadModels(from directory: URL, computeUnits: MLComputeUnits = .all) async throws {
        models = try await VibeVoiceAsrModels.load(from: directory, computeUnits: computeUnits)
        logger.info("VibeVoice-ASR models loaded successfully")
    }

    /// Whether all required models are loaded and ready.
    public var isAvailable: Bool {
        models != nil
    }

    /// Transcribe raw audio samples with unified ASR + diarization + timestamping.
    ///
    /// - Parameters:
    ///   - audioSamples: 24kHz mono Float32 audio samples. Audio at other sample rates
    ///                   should be resampled to 24kHz first.
    ///   - context: Optional context information (e.g., hotwords, speaker names, technical terms)
    ///              to improve transcription accuracy.
    ///   - maxNewTokens: Maximum number of tokens to generate.
    /// - Returns: Parsed transcription result with speaker-attributed, timestamped segments.
    public func transcribe(
        audioSamples: [Float],
        context: String? = nil,
        maxNewTokens: Int = VibeVoiceAsrConfig.defaultMaxNewTokens
    ) async throws -> VibeVoiceTranscriptionResult {
        guard let models = models else {
            throw VibeVoiceAsrError.generationFailed("Models not loaded. Call loadModels() first.")
        }

        guard !audioSamples.isEmpty else {
            throw VibeVoiceAsrError.audioTooShort
        }

        let audioDuration = Double(audioSamples.count) / Double(VibeVoiceAsrConfig.sampleRate)
        guard audioDuration <= VibeVoiceAsrConfig.maxAudioSeconds else {
            throw VibeVoiceAsrError.audioTooLong(audioDuration)
        }

        let start = CFAbsoluteTimeGetCurrent()

        // Step 1: Encode audio through acoustic tokenizer
        let t1 = CFAbsoluteTimeGetCurrent()
        let acousticFeatures = try encodeAcoustic(audioSamples: audioSamples, models: models)
        let acousticTime = CFAbsoluteTimeGetCurrent() - t1

        // Step 2: Encode audio through semantic tokenizer
        let t2 = CFAbsoluteTimeGetCurrent()
        let semanticFeatures = try encodeSemantic(audioSamples: audioSamples, models: models)
        let semanticTime = CFAbsoluteTimeGetCurrent() - t2

        // Step 3: Compute number of speech tokens
        let numSpeechTokens = Int(ceil(Double(audioSamples.count) / Double(VibeVoiceAsrConfig.compressionRatio)))

        // Step 4: Build prompt tokens and merge audio features
        let t4 = CFAbsoluteTimeGetCurrent()
        let promptTokens = buildPromptTokens(
            numSpeechTokens: numSpeechTokens,
            audioDuration: audioDuration,
            context: context
        )
        let initialEmbeddings = embedAndMerge(
            promptTokens: promptTokens,
            acousticFeatures: acousticFeatures,
            semanticFeatures: semanticFeatures,
            numSpeechTokens: numSpeechTokens,
            models: models
        )
        let embedTime = CFAbsoluteTimeGetCurrent() - t4

        // Step 5: Autoregressive generation
        // Explicitly release the previous KV-cache state before allocating the next one.
        decoderState = nil
        decoderState = models.decoderStateful.makeState()
        let t5 = CFAbsoluteTimeGetCurrent()
        let generatedTokenIds = try generate(
            initialEmbeddings: initialEmbeddings,
            promptLength: promptTokens.count,
            maxNewTokens: maxNewTokens,
            models: models,
            state: decoderState!
        )
        let generateTime = CFAbsoluteTimeGetCurrent() - t5

        // Step 6: Decode tokens to text
        let rawText = decodeTokens(generatedTokenIds, vocabulary: models.vocabulary)

        // Step 7: Parse structured output
        let result = VibeVoiceOutputParser.parse(rawText)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let rtfx = audioDuration / elapsed
        logger.info(
            "Transcribed \(String(format: "%.1f", audioDuration))s audio in \(String(format: "%.2f", elapsed))s (RTFx: \(String(format: "%.1f", rtfx))x) acoustic=\(String(format: "%.2f", acousticTime))s semantic=\(String(format: "%.2f", semanticTime))s embed=\(String(format: "%.2f", embedTime))s gen=\(String(format: "%.2f", generateTime))s tokens=\(generatedTokenIds.count) segments=\(result.segments.count)"
        )

        return result
    }

    // MARK: - Audio Encoding

    /// Encode audio through the acoustic tokenizer encoder.
    private func encodeAcoustic(
        audioSamples: [Float],
        models: VibeVoiceAsrModels
    ) throws -> [[Float]] {
        let audioInput = try createAudioInput(audioSamples: audioSamples, inputName: "audio")
        let prediction = try models.acousticEncoder.prediction(from: audioInput)

        guard let features = prediction.featureValue(for: "acoustic_features")?.multiArrayValue else {
            throw VibeVoiceAsrError.encoderFailed("No acoustic_features output")
        }

        return extractFeatures(from: features)
    }

    /// Encode audio through the semantic tokenizer encoder.
    private func encodeSemantic(
        audioSamples: [Float],
        models: VibeVoiceAsrModels
    ) throws -> [[Float]] {
        let audioInput = try createAudioInput(audioSamples: audioSamples, inputName: "audio")
        let prediction = try models.semanticEncoder.prediction(from: audioInput)

        guard let features = prediction.featureValue(for: "semantic_features")?.multiArrayValue else {
            throw VibeVoiceAsrError.encoderFailed("No semantic_features output")
        }

        return extractFeatures(from: features)
    }

    /// Create MLDictionaryFeatureProvider for audio input.
    private func createAudioInput(
        audioSamples: [Float],
        inputName: String
    ) throws -> MLDictionaryFeatureProvider {
        let shape: [NSNumber] = [1, NSNumber(value: audioSamples.count)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: audioSamples.count)

        audioSamples.withUnsafeBufferPointer { srcBuf in
            guard let srcPtr = srcBuf.baseAddress else { return }
            ptr.initialize(from: srcPtr, count: audioSamples.count)
        }

        return try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: array)])
    }

    /// Extract feature vectors from a CoreML MLMultiArray output.
    private func extractFeatures(from multiArray: MLMultiArray) -> [[Float]] {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count >= 2 else { return [] }

        let numFrames = shape[shape.count - 2]
        let featureDim = shape[shape.count - 1]
        var result: [[Float]] = []
        result.reserveCapacity(numFrames)

        for f in 0..<numFrames {
            var vec = [Float](repeating: 0.0, count: featureDim)
            for d in 0..<featureDim {
                let idx = f * featureDim + d
                vec[d] = multiArray[idx].floatValue
            }
            result.append(vec)
        }

        return result
    }

    // MARK: - Prompt Construction

    /// Build the chat template token sequence.
    ///
    /// Format follows the VibeVoice-ASR training template:
    /// ```
    /// <|im_start|>system\n{system_prompt}<|im_end|>\n
    /// <|im_start|>user\n<|speech_start|><|speech_pad|>...<|speech_end|>\n
    /// This is a {duration:.2f} seconds audio, please transcribe it with these keys: ...<|im_end|>\n
    /// <|im_start|>assistant\n
    /// ```
    private func buildPromptTokens(
        numSpeechTokens: Int,
        audioDuration: Double,
        context: String?
    ) -> [Int] {
        var tokens: [Int] = []

        // System message
        tokens.append(VibeVoiceAsrConfig.imStartTokenId)
        tokens.append(contentsOf: encodeText("system\n"))
        tokens.append(contentsOf: encodeText(VibeVoiceAsrConfig.systemPrompt))
        tokens.append(VibeVoiceAsrConfig.imEndTokenId)
        tokens.append(contentsOf: encodeText("\n"))

        // User message with speech tokens
        tokens.append(VibeVoiceAsrConfig.imStartTokenId)
        tokens.append(contentsOf: encodeText("user\n"))

        // Speech token sequence: <|speech_start|> + <|speech_pad|> * N + <|speech_end|>
        tokens.append(VibeVoiceAsrConfig.speechStartTokenId)
        for _ in 0..<numSpeechTokens {
            tokens.append(VibeVoiceAsrConfig.speechPadTokenId)
        }
        tokens.append(VibeVoiceAsrConfig.speechEndTokenId)

        // User suffix with audio metadata
        let keysString = VibeVoiceAsrConfig.outputKeys.joined(separator: ", ")
        let userSuffix: String
        if let ctx = context, !ctx.trimmingCharacters(in: .whitespaces).isEmpty {
            userSuffix =
                "\nThis is a \(String(format: "%.2f", audioDuration)) seconds audio, with extra info: "
                + "\(ctx.trimmingCharacters(in: .whitespaces))\n\nPlease transcribe it with these keys: \(keysString)"
        } else {
            userSuffix =
                "\nThis is a \(String(format: "%.2f", audioDuration)) seconds audio, please transcribe it with these keys: \(keysString)"
        }
        tokens.append(contentsOf: encodeText(userSuffix))
        tokens.append(VibeVoiceAsrConfig.imEndTokenId)
        tokens.append(contentsOf: encodeText("\n"))

        // Assistant start
        tokens.append(VibeVoiceAsrConfig.imStartTokenId)
        tokens.append(contentsOf: encodeText("assistant\n"))

        return tokens
    }

    /// Simple text to token encoding using the vocabulary.
    ///
    /// This is a simplified encoder for known prompt strings. For the chat template,
    /// most text consists of fixed strings with known token mappings.
    private func encodeText(_ text: String) -> [Int] {
        guard let models = models else { return [] }

        // Build reverse map for UTF-8-safe encoding
        // For the simple prompt strings, we can do character-by-character fallback
        var tokens: [Int] = []
        let vocab = models.vocabulary

        // Create reversed lookup: string → token ID
        var stringToId: [String: Int] = [:]
        for (id, str) in vocab {
            stringToId[str] = id
        }

        // Simple greedy longest-match encoding
        var remaining = text[...]
        while !remaining.isEmpty {
            var matched = false
            // Try increasingly shorter prefixes
            let maxLen = min(remaining.count, 32)
            for len in stride(from: maxLen, through: 1, by: -1) {
                let prefix = String(remaining.prefix(len))
                if let tokenId = stringToId[prefix] {
                    tokens.append(tokenId)
                    remaining = remaining.dropFirst(len)
                    matched = true
                    break
                }
            }
            if !matched {
                // Skip the character if we can't tokenize it
                remaining = remaining.dropFirst(1)
            }
        }

        return tokens
    }

    // MARK: - Feature Merging

    /// Embed prompt tokens and merge audio features at speech_pad positions.
    private func embedAndMerge(
        promptTokens: [Int],
        acousticFeatures: [[Float]],
        semanticFeatures: [[Float]],
        numSpeechTokens: Int,
        models: VibeVoiceAsrModels
    ) -> [[Float]] {
        let hiddenSize = models.hiddenSize
        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(promptTokens.count)

        var speechIdx = 0

        for token in promptTokens {
            if token == VibeVoiceAsrConfig.speechPadTokenId && speechIdx < numSpeechTokens {
                // Replace speech_pad token with merged audio features
                var merged = [Float](repeating: 0.0, count: hiddenSize)

                // Merge acoustic and semantic features
                // The exact projection depends on the model's internal projector weights,
                // which are part of the encoder outputs. Here we concatenate what's available.
                if speechIdx < acousticFeatures.count {
                    let acoustic = acousticFeatures[speechIdx]
                    for (i, val) in acoustic.enumerated() where i < hiddenSize {
                        merged[i] += val
                    }
                }
                if speechIdx < semanticFeatures.count {
                    let semantic = semanticFeatures[speechIdx]
                    for (i, val) in semantic.enumerated() where i < hiddenSize {
                        merged[i] += val
                    }
                }

                embeddings.append(merged)
                speechIdx += 1
            } else {
                // Regular token: look up embedding
                embeddings.append(models.embeddingWeights.embedding(for: token))
            }
        }

        return embeddings
    }

    // MARK: - Generation

    /// Autoregressive decode loop using the stateful decoder.
    private func generate(
        initialEmbeddings: [[Float]],
        promptLength: Int,
        maxNewTokens: Int,
        models: VibeVoiceAsrModels,
        state: MLState
    ) throws -> [Int] {
        let hiddenSize = models.hiddenSize
        let headDim = VibeVoiceAsrConfig.headDim
        let maxSeqLen = VibeVoiceAsrConfig.maxCacheSeqLen
        var generatedTokens: [Int] = []

        // Prefill: run all prompt embeddings through decoder
        let prefillLen = initialEmbeddings.count
        guard prefillLen > 0 else {
            throw VibeVoiceAsrError.generationFailed("No prompt embeddings")
        }

        // Process prefill in a single pass
        let prefillHidden = try createHiddenInput(initialEmbeddings, hiddenSize: hiddenSize)
        let prefillCos = try createRoPEInput(startPos: 0, length: prefillLen, headDim: headDim)
        let prefillSin = try createRoPEInput(startPos: 0, length: prefillLen, headDim: headDim, isSin: true)
        let prefillMask = try createCausalMask(queryLen: prefillLen, kvLen: prefillLen)

        let prefillInput = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: prefillHidden),
            "position_cos": MLFeatureValue(multiArray: prefillCos),
            "position_sin": MLFeatureValue(multiArray: prefillSin),
            "attention_mask": MLFeatureValue(multiArray: prefillMask),
        ])

        let prefillOutput = try models.decoderStateful.prediction(from: prefillInput, using: state)
        guard let logits = prefillOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw VibeVoiceAsrError.decoderFailed("No logits output from prefill")
        }

        // Get first token from prefill output
        var nextToken = argmax(logits: logits)
        generatedTokens.append(nextToken)

        // Decode loop
        var currentPos = prefillLen
        for _ in 1..<maxNewTokens {
            guard !VibeVoiceAsrConfig.eosTokenIds.contains(nextToken) else { break }
            guard currentPos < maxSeqLen else { break }

            // Get embedding for the generated token
            let tokenEmbedding = models.embeddingWeights.embedding(for: nextToken)

            // Create single-token inputs
            let hidden = try createHiddenInput([tokenEmbedding], hiddenSize: hiddenSize)
            let cos = try createRoPEInput(startPos: currentPos, length: 1, headDim: headDim)
            let sin = try createRoPEInput(startPos: currentPos, length: 1, headDim: headDim, isSin: true)
            let mask = try createCausalMask(queryLen: 1, kvLen: currentPos + 1)

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLFeatureValue(multiArray: hidden),
                "position_cos": MLFeatureValue(multiArray: cos),
                "position_sin": MLFeatureValue(multiArray: sin),
                "attention_mask": MLFeatureValue(multiArray: mask),
            ])

            let output = try models.decoderStateful.prediction(from: input, using: state)
            guard let stepLogits = output.featureValue(for: "logits")?.multiArrayValue else {
                throw VibeVoiceAsrError.decoderFailed("No logits output at step \(currentPos)")
            }

            nextToken = argmax(logits: stepLogits)
            generatedTokens.append(nextToken)
            currentPos += 1
        }

        return generatedTokens
    }

    // MARK: - Token Decoding

    /// Decode token IDs to text using the vocabulary.
    private func decodeTokens(_ tokenIds: [Int], vocabulary: [Int: String]) -> String {
        var result = ""
        for tokenId in tokenIds {
            guard !VibeVoiceAsrConfig.eosTokenIds.contains(tokenId) else { break }
            if let token = vocabulary[tokenId] {
                result += token
            }
        }
        return result
    }

    // MARK: - Helper Functions

    /// Create hidden_states MLMultiArray from embedding vectors.
    private func createHiddenInput(
        _ embeddings: [[Float]],
        hiddenSize: Int
    ) throws -> MLMultiArray {
        let seqLen = embeddings.count
        let shape: [NSNumber] = [1, NSNumber(value: seqLen), NSNumber(value: hiddenSize)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: seqLen * hiddenSize)

        for (i, embedding) in embeddings.enumerated() {
            let destOffset = i * hiddenSize
            let copyLen = min(embedding.count, hiddenSize)
            embedding.withUnsafeBufferPointer { srcBuf in
                guard let srcPtr = srcBuf.baseAddress else { return }
                ptr.advanced(by: destOffset).initialize(from: srcPtr, count: copyLen)
            }
            // Zero-fill if embedding is shorter than hiddenSize
            if copyLen < hiddenSize {
                ptr.advanced(by: destOffset + copyLen)
                    .initialize(repeating: 0.0, count: hiddenSize - copyLen)
            }
        }

        return array
    }

    /// Create RoPE position encoding arrays.
    private func createRoPEInput(
        startPos: Int,
        length: Int,
        headDim: Int,
        isSin: Bool = false
    ) throws -> MLMultiArray {
        let theta = VibeVoiceAsrConfig.ropeTheta
        let shape: [NSNumber] = [1, NSNumber(value: length), NSNumber(value: headDim)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: length * headDim)

        for pos in 0..<length {
            let absPos = startPos + pos
            for i in 0..<(headDim / 2) {
                let freq = 1.0 / pow(theta, Double(2 * i) / Double(headDim))
                let angle = Double(absPos) * freq
                let cosVal = Float(cos(angle))
                let sinVal = Float(sin(angle))

                let idx = pos * headDim
                if isSin {
                    ptr[idx + i] = sinVal
                    ptr[idx + headDim / 2 + i] = sinVal
                } else {
                    ptr[idx + i] = cosVal
                    ptr[idx + headDim / 2 + i] = cosVal
                }
            }
        }

        return array
    }

    /// Create causal attention mask.
    private func createCausalMask(queryLen: Int, kvLen: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: queryLen), NSNumber(value: kvLen)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: queryLen * kvLen)

        let negInf: Float = -1e9

        for q in 0..<queryLen {
            for k in 0..<kvLen {
                let idx = q * kvLen + k
                let queryPos = kvLen - queryLen + q
                ptr[idx] = k <= queryPos ? 0.0 : negInf
            }
        }

        return array
    }

    /// Argmax over logits to get the most likely token.
    private func argmax(logits: MLMultiArray) -> Int {
        let shape = logits.shape.map { $0.intValue }
        // logits shape: [1, 1, vocab_size] or [1, seq_len, vocab_size]
        let vocabSize = shape.last ?? 0
        guard vocabSize > 0 else { return 0 }

        // Get the last position's logits
        let totalElements = logits.count
        let startIdx = totalElements - vocabSize

        var maxVal: Float = -.infinity
        var maxIdx = 0

        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: totalElements)
        for i in 0..<vocabSize {
            let val = ptr[startIdx + i]
            if val > maxVal {
                maxVal = val
                maxIdx = i
            }
        }

        return maxIdx
    }
}

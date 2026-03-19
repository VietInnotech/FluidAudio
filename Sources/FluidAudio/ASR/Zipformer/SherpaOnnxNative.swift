#if os(macOS)
import CSherpaOnnx
import Foundation

enum SherpaOnnxNativeError: LocalizedError {
    case recognizerCreationFailed(String)
    case streamCreationFailed(String)
    case resultUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .recognizerCreationFailed(let message),
             .streamCreationFailed(let message),
             .resultUnavailable(let message):
            return message
        }
    }
}

enum SherpaOnnxProvider: String, Sendable {
    case cpu
    case coreml
}

struct SherpaOnnxOfflineTransducerPaths: Sendable {
    let encoder: String
    let decoder: String
    let joiner: String
    let tokens: String
}

struct SherpaOnnxOnlineTransducerPaths: Sendable {
    let encoder: String
    let decoder: String
    let joiner: String
    let tokens: String
}

private func sherpaCString(_ string: String) -> UnsafePointer<CChar>? {
    UnsafePointer((string as NSString).utf8String)
}

private func sherpaFeatureConfig(
    sampleRate: Int = 16_000,
    featureDim: Int = 80
) -> SherpaOnnxFeatureConfig {
    SherpaOnnxFeatureConfig(sample_rate: Int32(sampleRate), feature_dim: Int32(featureDim))
}

private func sherpaOfflineRecognizerConfig(
    paths: SherpaOnnxOfflineTransducerPaths,
    provider: SherpaOnnxProvider,
    numThreads: Int
) -> SherpaOnnxOfflineRecognizerConfig {
    let transducer = SherpaOnnxOfflineTransducerModelConfig(
        encoder: sherpaCString(paths.encoder),
        decoder: sherpaCString(paths.decoder),
        joiner: sherpaCString(paths.joiner)
    )

    let emptyParaformer = SherpaOnnxOfflineParaformerModelConfig(model: nil)
    let emptyNemo = SherpaOnnxOfflineNemoEncDecCtcModelConfig(model: nil)
    let emptyWhisper = SherpaOnnxOfflineWhisperModelConfig(
        encoder: nil,
        decoder: nil,
        language: nil,
        task: nil,
        tail_paddings: 0,
        enable_token_timestamps: 0,
        enable_segment_timestamps: 0
    )
    let emptyTdnn = SherpaOnnxOfflineTdnnModelConfig(model: nil)
    let emptySenseVoice = SherpaOnnxOfflineSenseVoiceModelConfig(model: nil, language: nil, use_itn: 0)
    let emptyMoonshine = SherpaOnnxOfflineMoonshineModelConfig(
        preprocessor: nil,
        encoder: nil,
        uncached_decoder: nil,
        cached_decoder: nil,
        merged_decoder: nil
    )
    let emptyFireRed = SherpaOnnxOfflineFireRedAsrModelConfig(encoder: nil, decoder: nil)
    let emptyDolphin = SherpaOnnxOfflineDolphinModelConfig(model: nil)
    let emptyZipformerCtc = SherpaOnnxOfflineZipformerCtcModelConfig(model: nil)
    let emptyCanary = SherpaOnnxOfflineCanaryModelConfig(
        encoder: nil,
        decoder: nil,
        src_lang: nil,
        tgt_lang: nil,
        use_pnc: 0
    )
    let emptyWenet = SherpaOnnxOfflineWenetCtcModelConfig(model: nil)
    let emptyOmnilingual = SherpaOnnxOfflineOmnilingualAsrCtcModelConfig(model: nil)
    let emptyMedAsr = SherpaOnnxOfflineMedAsrCtcModelConfig(model: nil)
    let emptyFunasr = SherpaOnnxOfflineFunASRNanoModelConfig(
        encoder_adaptor: nil,
        llm: nil,
        embedding: nil,
        tokenizer: nil,
        system_prompt: nil,
        user_prompt: nil,
        max_new_tokens: 0,
        temperature: 0,
        top_p: 0,
        seed: 0,
        language: nil,
        itn: 0,
        hotwords: nil
    )
    let emptyFireRedCtc = SherpaOnnxOfflineFireRedAsrCtcModelConfig(model: nil)
    let emptyLm = SherpaOnnxOfflineLMConfig(model: nil, scale: 0)
    let emptyHr = SherpaOnnxHomophoneReplacerConfig(dict_dir: nil, lexicon: nil, rule_fsts: nil)

    let modelConfig = SherpaOnnxOfflineModelConfig(
        transducer: transducer,
        paraformer: emptyParaformer,
        nemo_ctc: emptyNemo,
        whisper: emptyWhisper,
        tdnn: emptyTdnn,
        tokens: sherpaCString(paths.tokens),
        num_threads: Int32(numThreads),
        debug: 0,
        provider: sherpaCString(provider.rawValue),
        model_type: sherpaCString("transducer"),
        modeling_unit: sherpaCString("cjkchar"),
        bpe_vocab: sherpaCString(""),
        telespeech_ctc: sherpaCString(""),
        sense_voice: emptySenseVoice,
        moonshine: emptyMoonshine,
        fire_red_asr: emptyFireRed,
        dolphin: emptyDolphin,
        zipformer_ctc: emptyZipformerCtc,
        canary: emptyCanary,
        wenet_ctc: emptyWenet,
        omnilingual: emptyOmnilingual,
        medasr: emptyMedAsr,
        funasr_nano: emptyFunasr,
        fire_red_asr_ctc: emptyFireRedCtc
    )

    return SherpaOnnxOfflineRecognizerConfig(
        feat_config: sherpaFeatureConfig(),
        model_config: modelConfig,
        lm_config: emptyLm,
        decoding_method: sherpaCString("greedy_search"),
        max_active_paths: 4,
        hotwords_file: sherpaCString(""),
        hotwords_score: 1.5,
        rule_fsts: sherpaCString(""),
        rule_fars: sherpaCString(""),
        blank_penalty: 0,
        hr: emptyHr
    )
}

private func sherpaOnlineRecognizerConfig(
    paths: SherpaOnnxOnlineTransducerPaths,
    provider: SherpaOnnxProvider,
    numThreads: Int
) -> SherpaOnnxOnlineRecognizerConfig {
    let transducer = SherpaOnnxOnlineTransducerModelConfig(
        encoder: sherpaCString(paths.encoder),
        decoder: sherpaCString(paths.decoder),
        joiner: sherpaCString(paths.joiner)
    )

    let emptyParaformer = SherpaOnnxOnlineParaformerModelConfig(encoder: nil, decoder: nil)
    let emptyZipformer2Ctc = SherpaOnnxOnlineZipformer2CtcModelConfig(model: nil)
    let emptyNemo = SherpaOnnxOnlineNemoCtcModelConfig(model: nil)
    let emptyTone = SherpaOnnxOnlineToneCtcModelConfig(model: nil)
    let emptyCtcFst = SherpaOnnxOnlineCtcFstDecoderConfig(graph: nil, max_active: 0)
    let emptyHr = SherpaOnnxHomophoneReplacerConfig(dict_dir: nil, lexicon: nil, rule_fsts: nil)

    let modelConfig = SherpaOnnxOnlineModelConfig(
        transducer: transducer,
        paraformer: emptyParaformer,
        zipformer2_ctc: emptyZipformer2Ctc,
        tokens: sherpaCString(paths.tokens),
        num_threads: Int32(numThreads),
        provider: sherpaCString(provider.rawValue),
        debug: 0,
        model_type: sherpaCString(""),
        modeling_unit: sherpaCString("cjkchar"),
        bpe_vocab: sherpaCString(""),
        tokens_buf: sherpaCString(""),
        tokens_buf_size: 0,
        nemo_ctc: emptyNemo,
        t_one_ctc: emptyTone
    )

    return SherpaOnnxOnlineRecognizerConfig(
        feat_config: sherpaFeatureConfig(),
        model_config: modelConfig,
        decoding_method: sherpaCString("greedy_search"),
        max_active_paths: 4,
        enable_endpoint: 0,
        rule1_min_trailing_silence: 2.4,
        rule2_min_trailing_silence: 1.2,
        rule3_min_utterance_length: 20.0,
        hotwords_file: sherpaCString(""),
        hotwords_score: 1.5,
        ctc_fst_decoder_config: emptyCtcFst,
        rule_fsts: sherpaCString(""),
        rule_fars: sherpaCString(""),
        blank_penalty: 0,
        hotwords_buf: sherpaCString(""),
        hotwords_buf_size: 0,
        hr: emptyHr
    )
}

final class SherpaOnnxOfflineZipformerRecognizer {
    private let recognizer: OpaquePointer

    init(paths: SherpaOnnxOfflineTransducerPaths, provider: SherpaOnnxProvider, numThreads: Int = 1) throws {
        var config = sherpaOfflineRecognizerConfig(paths: paths, provider: provider, numThreads: numThreads)
        guard let recognizer = SherpaOnnxCreateOfflineRecognizer(&config) else {
            throw SherpaOnnxNativeError.recognizerCreationFailed(
                "Failed to create sherpa-onnx offline recognizer for \(paths.encoder)"
            )
        }
        self.recognizer = recognizer
    }

    deinit {
        SherpaOnnxDestroyOfflineRecognizer(recognizer)
    }

    func decode(samples: [Float], sampleRate: Int = 16_000) throws -> String {
        guard let stream = SherpaOnnxCreateOfflineStream(recognizer) else {
            throw SherpaOnnxNativeError.streamCreationFailed("Failed to create sherpa-onnx offline stream")
        }
        defer { SherpaOnnxDestroyOfflineStream(stream) }

        SherpaOnnxAcceptWaveformOffline(stream, Int32(sampleRate), samples, Int32(samples.count))
        SherpaOnnxDecodeOfflineStream(recognizer, stream)

        guard let result = SherpaOnnxGetOfflineStreamResult(stream) else {
            throw SherpaOnnxNativeError.resultUnavailable("Failed to read sherpa-onnx offline result")
        }
        defer { SherpaOnnxDestroyOfflineRecognizerResult(result) }

        guard let text = result.pointee.text else {
            return ""
        }

        return String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

final class SherpaOnnxStreamingZipformerRecognizer {
    private let recognizer: OpaquePointer

    init(paths: SherpaOnnxOnlineTransducerPaths, provider: SherpaOnnxProvider, numThreads: Int = 1) throws {
        var config = sherpaOnlineRecognizerConfig(paths: paths, provider: provider, numThreads: numThreads)
        guard let recognizer = SherpaOnnxCreateOnlineRecognizer(&config) else {
            throw SherpaOnnxNativeError.recognizerCreationFailed(
                "Failed to create sherpa-onnx streaming recognizer for \(paths.encoder)"
            )
        }
        self.recognizer = recognizer
    }

    deinit {
        SherpaOnnxDestroyOnlineRecognizer(recognizer)
    }

    func decode(samples: [Float], sampleRate: Int = 16_000, tailPaddingSeconds: Double = 0.3) throws -> String {
        guard let stream = SherpaOnnxCreateOnlineStream(recognizer) else {
            throw SherpaOnnxNativeError.streamCreationFailed("Failed to create sherpa-onnx streaming stream")
        }
        defer { SherpaOnnxDestroyOnlineStream(stream) }

        SherpaOnnxOnlineStreamAcceptWaveform(stream, Int32(sampleRate), samples, Int32(samples.count))

        if tailPaddingSeconds > 0 {
            let tailSamples = Int(Double(sampleRate) * tailPaddingSeconds)
            let padding = [Float](repeating: 0, count: tailSamples)
            SherpaOnnxOnlineStreamAcceptWaveform(stream, Int32(sampleRate), padding, Int32(padding.count))
        }

        SherpaOnnxOnlineStreamInputFinished(stream)

        while SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
            SherpaOnnxDecodeOnlineStream(recognizer, stream)
        }

        guard let result = SherpaOnnxGetOnlineStreamResult(recognizer, stream) else {
            throw SherpaOnnxNativeError.resultUnavailable("Failed to read sherpa-onnx streaming result")
        }
        defer { SherpaOnnxDestroyOnlineRecognizerResult(result) }

        guard let text = result.pointee.text else {
            return ""
        }

        return String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
#endif

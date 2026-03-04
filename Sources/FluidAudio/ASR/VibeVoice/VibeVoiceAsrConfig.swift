import Foundation

// MARK: - VibeVoice-ASR Model Configuration

/// Configuration constants for the VibeVoice-ASR CoreML model.
///
/// Architecture: acoustic_encoder + semantic_encoder → embedding merge → Qwen2.5-7B decoder → structured JSON
/// VibeVoice-ASR is a unified model that jointly performs ASR, speaker diarization, and timestamping.
///
/// Reference: https://arxiv.org/abs/2601.18184
/// Model: https://huggingface.co/microsoft/VibeVoice-ASR
@available(macOS 15, iOS 18, *)
public enum VibeVoiceAsrConfig {
    // MARK: Audio

    /// VibeVoice uses 24kHz audio (unlike Parakeet/Qwen3 which use 16kHz).
    public static let sampleRate = 24000

    /// Acoustic/semantic tokenizer compression ratio: 3200x at 24kHz.
    /// This means ~7.5 tokens per second of audio.
    public static let compressionRatio = 3200

    /// Maximum audio duration supported (60 minutes).
    public static let maxAudioSeconds: Double = 3600.0

    // MARK: Acoustic Tokenizer

    /// Acoustic tokenizer VAE latent dimension.
    public static let acousticVaeDim = 64

    /// Acoustic tokenizer encoder downsampling ratios: [8, 5, 5, 4, 2, 2] = 3200x.
    public static let acousticDownsampleRatios = [8, 5, 5, 4, 2, 2]

    // MARK: Semantic Tokenizer

    /// Semantic tokenizer VAE latent dimension.
    public static let semanticVaeDim = 128

    // MARK: Decoder (Qwen2.5-7B)

    public static let hiddenSize = 3584
    public static let intermediateSize = 18944
    public static let numDecoderLayers = 28
    public static let numAttentionHeads = 28
    public static let numKVHeads = 4
    public static let headDim = 128
    public static let vocabSize = 152_064
    public static let ropeTheta: Double = 1_000_000
    public static let rmsNormEps: Double = 1e-6

    // MARK: Generation

    /// Maximum KV cache sequence length for the stateful decoder model.
    /// Must match the value used during CoreML conversion.
    public static let maxCacheSeqLen = 4096

    /// Default maximum number of new tokens to generate.
    public static let defaultMaxNewTokens = 8192

    // MARK: Special Tokens

    /// `<|speech_start|>` — marks the beginning of audio features.
    public static let speechStartTokenId = 151_852

    /// `<|speech_end|>` — marks the end of audio features.
    public static let speechEndTokenId = 151_853

    /// `<|speech_pad|>` — placeholder for each speech token (replaced by audio features).
    public static let speechPadTokenId = 151_854

    /// `<|endoftext|>` — end of text / padding token.
    public static let eotTokenId = 151_643

    /// EOS token IDs that terminate generation.
    public static let eosTokenIds: Set<Int> = [151_643, 151_645]

    // MARK: Chat Template Tokens

    /// `<|im_start|>` — chat template role start.
    public static let imStartTokenId = 151_644

    /// `<|im_end|>` — chat template role end.
    public static let imEndTokenId = 151_645

    // MARK: System Prompt

    /// The system prompt used during VibeVoice-ASR training.
    public static let systemPrompt =
        "You are a helpful assistant that transcribes audio input into text output in JSON format."

    /// The transcription keys requested in the user prompt.
    public static let outputKeys = ["Start time", "End time", "Speaker ID", "Content"]

    // MARK: - Supported Languages

    /// VibeVoice-ASR supports 50+ languages with no explicit language setting required.
    /// Language detection is automatic. This enum provides common language codes for
    /// use in context/hotword prompts when needed.
    public enum Language: String, CaseIterable, Sendable {
        case english = "en"
        case chinese = "zh"
        case french = "fr"
        case german = "de"
        case spanish = "es"
        case italian = "it"
        case japanese = "ja"
        case korean = "ko"
        case russian = "ru"
        case portuguese = "pt"
        case arabic = "ar"
        case hindi = "hi"
        case thai = "th"
        case vietnamese = "vi"
        case indonesian = "id"
        case turkish = "tr"
        case dutch = "nl"
        case swedish = "sv"
        case danish = "da"
        case finnish = "fi"
        case polish = "pl"
        case czech = "cs"

        /// English name for the language.
        public var englishName: String {
            switch self {
            case .english: return "English"
            case .chinese: return "Chinese"
            case .french: return "French"
            case .german: return "German"
            case .spanish: return "Spanish"
            case .italian: return "Italian"
            case .japanese: return "Japanese"
            case .korean: return "Korean"
            case .russian: return "Russian"
            case .portuguese: return "Portuguese"
            case .arabic: return "Arabic"
            case .hindi: return "Hindi"
            case .thai: return "Thai"
            case .vietnamese: return "Vietnamese"
            case .indonesian: return "Indonesian"
            case .turkish: return "Turkish"
            case .dutch: return "Dutch"
            case .swedish: return "Swedish"
            case .danish: return "Danish"
            case .finnish: return "Finnish"
            case .polish: return "Polish"
            case .czech: return "Czech"
            }
        }

        /// Initialize from ISO code or English name.
        public init?(from string: String) {
            let lowercased = string.lowercased()
            if let lang = Language(rawValue: lowercased) {
                self = lang
                return
            }
            if let lang = Language.allCases.first(where: { $0.englishName.lowercased() == lowercased }) {
                self = lang
                return
            }
            return nil
        }
    }
}

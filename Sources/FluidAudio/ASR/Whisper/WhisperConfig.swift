import Foundation

// MARK: - Whisper Configuration

/// Configuration constants for Whisper Large v3 Turbo model.
/// Adapted from WhisperKit (MIT License, Copyright © 2024 Argmax, Inc.)
public enum WhisperConfig {

    // MARK: - Model Dimensions

    /// Model embedding dimension
    public static let dModel = 1280
    /// Number of decoder attention heads
    public static let decoderAttentionHeads = 20
    /// Vocabulary size
    public static let vocabSize = 51866
    /// Number of mel filter banks
    public static let numMelBins = 128
    /// Maximum encoder source positions (frames)
    public static let maxSourcePositions = 1500
    /// Maximum decoder target positions (tokens)
    public static let maxTargetPositions = 448
    /// Maximum token context for KV cache (half of maxTargetPositions)
    public static let maxTokenContext = 224
    /// KV cache embed dimension (stacked across all decoder layers)
    public static let kvCacheEmbedDim = 40960
    /// Audio sample rate
    public static let sampleRate = 16000
    /// Samples per 30-second window
    public static let windowSamples = 480_000
    /// Seconds per time token (Whisper uses 0.02s per time token)
    public static let secondsPerTimeToken: Float = 0.02

    // MARK: - Special Token IDs

    public enum Tokens {
        public static let endOfText = 50257
        public static let startOfTranscript = 50258
        public static let english = 50259
        public static let transcribe = 50360
        public static let translate = 50359
        public static let noTimestamps = 50364
        public static let timeTokenBegin = 50364
        public static let startOfPrevious = 50361
        public static let noSpeech = 50362
        public static let specialTokenBegin = 50257
        public static let whitespace = 220  // " " token
    }

    // MARK: - Suppress Tokens (from generation_config.json)

    /// Tokens to suppress during decoding (from generation_config.json suppress_tokens)
    public static let suppressTokens: [Int] = [
        1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
        90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350,
        1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
        6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
        13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
        21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
        36865, 42863, 47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363,
    ]

    // MARK: - Language Mapping

    /// Language code to token ID mapping (from generation_config.json lang_to_id)
    public static let languageToTokenId: [String: Int] = [
        "af": 50327, "am": 50334, "ar": 50272, "as": 50350, "az": 50304,
        "ba": 50355, "be": 50330, "bg": 50292, "bn": 50302, "bo": 50347,
        "br": 50309, "bs": 50315, "ca": 50270, "cs": 50283, "cy": 50297,
        "da": 50285, "de": 50261, "el": 50281, "en": 50259, "es": 50262,
        "et": 50307, "eu": 50310, "fa": 50300, "fi": 50277, "fo": 50338,
        "fr": 50265, "gl": 50319, "gu": 50333, "ha": 50354, "haw": 50352,
        "he": 50279, "hi": 50276, "hr": 50291, "ht": 50339, "hu": 50286,
        "hy": 50312, "id": 50275, "is": 50311, "it": 50274, "ja": 50266,
        "jw": 50356, "ka": 50329, "kk": 50316, "km": 50323, "kn": 50306,
        "ko": 50264, "la": 50294, "lb": 50345, "ln": 50353, "lo": 50336,
        "lt": 50293, "lv": 50301, "mg": 50349, "mi": 50295, "mk": 50308,
        "ml": 50296, "mn": 50314, "mr": 50320, "ms": 50282, "mt": 50343,
        "my": 50346, "ne": 50313, "nl": 50271, "nn": 50342, "no": 50288,
        "oc": 50328, "pa": 50321, "pl": 50269, "ps": 50340, "pt": 50267,
        "ro": 50284, "ru": 50263, "sa": 50344, "sd": 50332, "si": 50322,
        "sk": 50298, "sl": 50305, "sn": 50324, "so": 50326, "sq": 50317,
        "sr": 50303, "su": 50357, "sv": 50273, "sw": 50318, "ta": 50287,
        "te": 50299, "tg": 50331, "th": 50289, "tk": 50341, "tl": 50348,
        "tr": 50268, "tt": 50351, "uk": 50280, "ur": 50290, "uz": 50337,
        "vi": 50278, "yi": 50335, "yo": 50325, "yue": 50358, "zh": 50260,
    ]

    /// Default language code
    public static let defaultLanguageCode = "en"

    /// Resolve a language string to a token ID. Returns english token if not found.
    public static func languageTokenId(for language: String?) -> Int {
        guard let lang = language else { return Tokens.english }
        return languageToTokenId[lang] ?? Tokens.english
    }
}

// MARK: - Decoding Options

/// Options for Whisper decoding.
public struct WhisperDecodingOptions: Sendable {
    /// Language code (e.g. "en", "fr"). nil for auto-detect.
    public var language: String?
    /// Decoding task
    public var task: WhisperTask
    /// Temperature for sampling (0 = greedy)
    public var temperature: Float
    /// Whether to suppress timestamps
    public var withoutTimestamps: Bool
    /// Whether to use the prefill cache model
    public var usePrefillCache: Bool
    /// Maximum number of tokens to decode per 30s window
    public var sampleLength: Int
    /// Whether to suppress blank tokens at start
    public var suppressBlank: Bool
    /// First token log probability threshold for fallback
    public var firstTokenLogProbThreshold: Float?
    /// Compression ratio threshold for fallback
    public var compressionRatioThreshold: Float?
    /// Average log probability threshold for fallback
    public var logProbThreshold: Float?

    public init(
        language: String? = "en",
        task: WhisperTask = .transcribe,
        temperature: Float = 0,
        withoutTimestamps: Bool = true,
        usePrefillCache: Bool = true,
        sampleLength: Int = WhisperConfig.maxTokenContext,
        suppressBlank: Bool = true,
        firstTokenLogProbThreshold: Float? = -1.5,
        compressionRatioThreshold: Float? = 2.4,
        logProbThreshold: Float? = -1.0
    ) {
        self.language = language
        self.task = task
        self.temperature = temperature
        self.withoutTimestamps = withoutTimestamps
        self.usePrefillCache = usePrefillCache
        self.sampleLength = sampleLength
        self.suppressBlank = suppressBlank
        self.firstTokenLogProbThreshold = firstTokenLogProbThreshold
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logProbThreshold = logProbThreshold
    }
}

/// Whisper task types.
public enum WhisperTask: String, Codable, Sendable {
    case transcribe
    case translate
}

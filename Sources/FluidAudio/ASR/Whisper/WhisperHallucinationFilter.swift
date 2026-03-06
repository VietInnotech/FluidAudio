import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "WhisperHallucinationFilter")

// MARK: - Whisper Hallucination Filter

/// Filters common Whisper hallucinations from transcription output.
///
/// Whisper models are prone to generating hallucinated text, especially:
/// - During silence or low-energy audio segments
/// - Repetitive phrases from training data (e.g. YouTube subscribe prompts)
/// - Text with abnormally high compression ratios (repetition)
/// - Text with abnormally low average log-probabilities (low confidence)
///
/// This filter applies three layers of detection:
/// 1. **Known phrases**: Exact/substring match against a curated list of common hallucinations
/// 2. **Compression ratio**: Rejects windows with excessive token repetition
/// 3. **Log probability**: Rejects windows where the model is not confident
public enum WhisperHallucinationFilter {

    // MARK: - Known Hallucination Phrases

    /// Common hallucinated phrases across multiple languages.
    /// These are substring patterns — if the decoded text contains any of them,
    /// the entire window is flagged as a hallucination.
    public static let knownHallucinationPhrases: [String] = [
        // Vietnamese
        "Hãy subscribe cho kênh",
        "Ghiền Mì Gõ",
        "Để không bỏ lỡ những video hấp dẫn",
        "Cảm ơn các bạn đã xem video",
        "Hãy like share và subscribe",
        "Nhớ like share và đăng ký kênh",
        "Hãy đăng ký kênh",
        "Hẹn gặp lại các bạn trong những video tiếp theo.",
        "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn",
        "Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ những video hấp dẫn",
        "Hãy subscribe cho kênh Ghiền Mì Gõi",
        "Hãy subscribe cho kênh lalaschool Để không bỏ lỡ những video hấp dẫn",
        "Để không bỏ lỡ những video hấp dẫn",
        "Cảm ơn các bạn đã theo dõi và hẹn gặp lại.",
        "Cảm ơn các bạn đã theo dõi.",
        "Để không bỏ lỡ những gì hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn."

        // English
        "Thank you for watching",
        "Please subscribe to my channel",
        "Thanks for watching",
        "Please like and subscribe",
        "Don't forget to subscribe",
        "Subscribe and hit the bell",
        "Like, share, and subscribe",
        "If you enjoyed this video",
        "See you in the next video",
        "Check out my other videos",

        // Chinese
        "请订阅我的频道",
        "感谢收看",
        "别忘了点赞",
        "欢迎订阅",
        "谢谢观看",
        "请点赞订阅",

        // Japanese
        "チャンネル登録お願いします",
        "ご視聴ありがとうございました",
        "チャンネル登録よろしく",
        "高評価よろしくお願いします",

        // Korean
        "구독과 좋아요 부탁드립니다",
        "시청해주셔서 감사합니다",
        "구독 좋아요 알림설정",

        // Spanish
        "Suscríbete al canal",
        "Gracias por ver el video",
        "No olvides suscribirte",
        "Dale like y suscríbete",

        // French
        "Abonnez-vous à la chaîne",
        "Merci d'avoir regardé",
        "N'oubliez pas de vous abonner",

        // Portuguese
        "Inscreva-se no canal",
        "Obrigado por assistir",
        "Não esqueça de se inscrever",
        "Deixe o like e se inscreva",

        // German
        "Abonniert den Kanal",
        "Danke fürs Zuschauen",
        "Vergesst nicht zu abonnieren",

        // Generic patterns (any language)
        "www.",
        "http://",
        "https://",
        ".com",
    ]

    /// Short phrases that are only flagged as hallucinations when they appear
    /// as the entire (or near-entire) output of a window. These are common
    /// greetings/closings that Whisper emits over silence.
    public static let shortHallucinationPhrases: [String] = [
        "...",
        "♪",
        "♫",
        "Sottotitoli creati dalla comunità Amara.org",
        "Sous-titres réalisés para la communauté d'Amara.org",
        "ご視聴ありがとうございました",
    ]

    // MARK: - Thresholds

    /// Default compression ratio threshold. Windows above this are likely hallucinated repetition.
    /// Whisper's original default is 2.4; we use a slightly tighter threshold for filtering.
    public static let defaultCompressionRatioThreshold: Float = 2.4

    /// Default average log probability threshold. Windows below this have low model confidence.
    /// Whisper's original default is -1.0.
    public static let defaultLogProbThreshold: Float = -1.0

    // MARK: - Public API

    /// Check whether a decoded window result is likely a hallucination.
    ///
    /// - Parameters:
    ///   - result: The decoded result from a single 30s window.
    ///   - compressionRatioThreshold: Max allowed compression ratio (default 2.4).
    ///   - logProbThreshold: Min allowed average log probability (default -1.0).
    /// - Returns: `true` if the result should be discarded as a hallucination.
    public static func isHallucination(
        _ result: WhisperDecodingResult,
        compressionRatioThreshold: Float = defaultCompressionRatioThreshold,
        logProbThreshold: Float = defaultLogProbThreshold
    ) -> Bool {
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else { return false }

        // Check known hallucination phrases
        if containsKnownHallucination(text) {
            logger.info("Hallucination detected (known phrase): \"\(text.prefix(80))\"")
            return true
        }

        // Check short hallucination phrases (only flag if text is very short)
        if text.count < 40, containsShortHallucination(text) {
            logger.info("Hallucination detected (short phrase): \"\(text)\"")
            return true
        }

        // Check compression ratio (high = repetitive tokens)
        if result.compressionRatio > compressionRatioThreshold {
            logger.info(
                "Hallucination detected (compression ratio \(String(format: "%.2f", result.compressionRatio)) > \(String(format: "%.2f", compressionRatioThreshold))): \"\(text.prefix(80))\""
            )
            return true
        }

        // Check average log probability (low = model not confident)
        if result.avgLogProb < logProbThreshold {
            logger.info(
                "Hallucination detected (avg log prob \(String(format: "%.2f", result.avgLogProb)) < \(String(format: "%.2f", logProbThreshold))): \"\(text.prefix(80))\""
            )
            return true
        }

        return false
    }

    /// Remove known hallucination substrings from transcribed text.
    ///
    /// Unlike `isHallucination` (which discards entire windows), this performs
    /// fine-grained removal of hallucinated substrings within otherwise valid text.
    /// Useful as a post-processing step on the final joined transcript.
    ///
    /// - Parameter text: The full transcription text.
    /// - Returns: Cleaned text with hallucinated substrings removed.
    public static func removeHallucinationSubstrings(_ text: String) -> String {
        var cleaned = text

        for phrase in knownHallucinationPhrases {
            // Skip very short generic patterns for substring removal
            // (they could match legitimate content)
            guard phrase.count > 8 else { continue }
            cleaned = cleaned.replacingOccurrences(of: phrase, with: "")
        }

        // Collapse multiple spaces and trim
        cleaned = cleaned.replacingOccurrences(
            of: "\\s{2,}",
            with: " ",
            options: .regularExpression
        )
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Private Helpers

    private static func containsKnownHallucination(_ text: String) -> Bool {
        let lowercased = text.lowercased()
        for phrase in knownHallucinationPhrases {
            guard phrase.count > 8 else { continue }  // Skip URL-like short patterns for this check
            if lowercased.contains(phrase.lowercased()) {
                return true
            }
        }
        return false
    }

    private static func containsShortHallucination(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for phrase in shortHallucinationPhrases {
            if trimmed.lowercased() == phrase.lowercased()
                || trimmed.lowercased().hasPrefix(phrase.lowercased())
            {
                return true
            }
        }
        return false
    }
}

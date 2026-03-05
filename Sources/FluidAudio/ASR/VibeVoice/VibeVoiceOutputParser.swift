import Foundation

// MARK: - VibeVoice-ASR Output Parser

/// A parsed transcription segment from VibeVoice-ASR output.
///
/// VibeVoice-ASR outputs structured JSON containing "Who" (speaker), "When" (timestamps),
/// and "What" (content) for each segment of speech.
public struct VibeVoiceTranscriptionSegment: Sendable, Codable, Equatable {
    /// Start time of the segment (e.g., "0.00s").
    public let startTime: String

    /// End time of the segment (e.g., "2.50s").
    public let endTime: String

    public init(startTime: String, endTime: String, speakerId: String, content: String) {
        self.startTime = startTime
        self.endTime = endTime
        self.speakerId = speakerId
        self.content = content
    }

    /// Speaker identifier (e.g., "Speaker 1").
    public let speakerId: String

    /// Transcribed text content.
    public let content: String

    /// Start time in seconds (parsed from startTime string).
    public var startTimeSeconds: Double? {
        parseTimeString(startTime)
    }

    /// End time in seconds (parsed from endTime string).
    public var endTimeSeconds: Double? {
        parseTimeString(endTime)
    }

    private func parseTimeString(_ timeStr: String) -> Double? {
        let cleaned = timeStr.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "s", with: "")
        return Double(cleaned)
    }

    enum CodingKeys: String, CodingKey {
        case startTime = "Start time"
        case endTime = "End time"
        case speakerId = "Speaker ID"
        case content = "Content"
    }
}

/// Complete transcription result from VibeVoice-ASR.
public struct VibeVoiceTranscriptionResult: Sendable {
    /// The parsed transcription segments with speaker IDs, timestamps, and content.
    public let segments: [VibeVoiceTranscriptionSegment]

    /// The raw generated text before parsing.
    public let rawText: String

    public init(segments: [VibeVoiceTranscriptionSegment], rawText: String) {
        self.segments = segments
        self.rawText = rawText
    }

    /// Number of unique speakers detected.
    public var speakerCount: Int {
        Set(segments.map(\.speakerId)).count
    }

    /// Total duration covered by the transcription (seconds).
    public var totalDuration: Double? {
        guard let lastEnd = segments.last?.endTimeSeconds else { return nil }
        return lastEnd
    }

    /// Plain text transcription without speaker/timestamp annotations.
    public var plainText: String {
        segments.map(\.content).joined(separator: " ")
    }
}

// MARK: - Parser

/// Parses VibeVoice-ASR structured output into typed segments.
///
/// VibeVoice-ASR generates JSON output in the following format:
/// ```json
/// [
///   {"Start time": "0.00s", "End time": "2.50s", "Speaker ID": "Speaker 1", "Content": "Hello."},
///   {"Start time": "2.80s", "End time": "5.10s", "Speaker ID": "Speaker 2", "Content": "Good morning."}
/// ]
/// ```
///
/// The parser handles common variations:
/// - JSON wrapped in markdown code blocks (```json ... ```)
/// - Partial JSON at the end (truncated generation)
/// - Whitespace/formatting differences
public enum VibeVoiceOutputParser {

    /// Parse the raw generated text into a transcription result.
    ///
    /// - Parameter text: Raw text output from the model.
    /// - Returns: Parsed transcription result with segments.
    public static func parse(_ text: String) -> VibeVoiceTranscriptionResult {
        let segments = parseSegments(text)
        return VibeVoiceTranscriptionResult(segments: segments, rawText: text)
    }

    /// Parse structured JSON output into transcription segments.
    ///
    /// - Parameter text: Raw text output from the model.
    /// - Returns: Array of parsed segments, or empty array if parsing fails.
    public static func parseSegments(_ text: String) -> [VibeVoiceTranscriptionSegment] {
        let jsonString = extractJSON(from: text)
        guard !jsonString.isEmpty else { return [] }

        let decoder = JSONDecoder()

        // Try parsing as a JSON array of segments
        if let data = jsonString.data(using: .utf8),
            let segments = try? decoder.decode([VibeVoiceTranscriptionSegment].self, from: data)
        {
            return segments
        }

        // Try parsing as a single segment (model sometimes returns a single object)
        if let data = jsonString.data(using: .utf8),
            let segment = try? decoder.decode(VibeVoiceTranscriptionSegment.self, from: data)
        {
            return [segment]
        }

        // Try repairing truncated JSON (missing closing brackets)
        let repaired = repairTruncatedJSON(jsonString)
        if let data = repaired.data(using: .utf8),
            let segments = try? decoder.decode([VibeVoiceTranscriptionSegment].self, from: data)
        {
            return segments
        }

        return []
    }

    // MARK: - Private Helpers

    /// Extract JSON content from the generated text, handling markdown code blocks.
    private static func extractJSON(from text: String) -> String {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Handle markdown code blocks: ```json ... ```
        if trimmed.contains("```json") {
            guard let jsonStart = trimmed.range(of: "```json") else { return trimmed }
            trimmed = String(trimmed[jsonStart.upperBound...])
            if let jsonEnd = trimmed.range(of: "```") {
                trimmed = String(trimmed[..<jsonEnd.lowerBound])
            }
            return trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Handle plain code blocks: ``` ... ```
        if trimmed.hasPrefix("```") {
            trimmed = String(trimmed.dropFirst(3))
            if let jsonEnd = trimmed.range(of: "```") {
                trimmed = String(trimmed[..<jsonEnd.lowerBound])
            }
            return trimmed.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Find the first JSON array or object
        guard let firstBracket = trimmed.firstIndex(where: { $0 == "[" || $0 == "{" }) else {
            return trimmed
        }

        return String(trimmed[firstBracket...])
    }

    /// Attempt to repair truncated JSON by closing unclosed brackets/braces.
    private static func repairTruncatedJSON(_ json: String) -> String {
        var result = json.trimmingCharacters(in: .whitespacesAndNewlines)

        // Remove trailing comma if present
        if result.hasSuffix(",") {
            result = String(result.dropLast())
        }

        // Count unmatched brackets
        var openBraces = 0
        var openBrackets = 0
        var inString = false
        var prevChar: Character = " "

        for char in result {
            if char == "\"" && prevChar != "\\" {
                inString.toggle()
            }
            guard !inString else {
                prevChar = char
                continue
            }
            switch char {
            case "{": openBraces += 1
            case "}": openBraces -= 1
            case "[": openBrackets += 1
            case "]": openBrackets -= 1
            default: break
            }
            prevChar = char
        }

        // Close unclosed structures
        for _ in 0..<max(0, openBraces) {
            result += "}"
        }
        for _ in 0..<max(0, openBrackets) {
            result += "]"
        }

        return result
    }
}

import Foundation

enum ZipformerVnTranscriptPostprocessor {
    private static let removedChars: Set<Character> = [
        "\u{FEFF}",
        "\u{200B}",
        "\u{200C}",
        "\u{200D}",
        "\u{2060}",
    ]

    static func cleanSegmentText(_ text: String) -> String {
        normalizeText(text, lowercase: false)
    }

    static func combineSegmentTexts(_ texts: [String], maxOverlapWords: Int = 12) -> String {
        var mergedDisplayWords: [String] = []
        var mergedNormalizedWords: [String] = []

        for text in texts {
            let cleaned = cleanSegmentText(text)
            guard !cleaned.isEmpty else {
                continue
            }

            let displayWords = cleaned.split(whereSeparator: \.isWhitespace).map(String.init)
            let normalizedWords = displayWords.map { normalizeText($0, lowercase: true) }.filter { !$0.isEmpty }

            guard !displayWords.isEmpty, displayWords.count == normalizedWords.count else {
                continue
            }

            if mergedDisplayWords.isEmpty {
                mergedDisplayWords = displayWords
                mergedNormalizedWords = normalizedWords
                continue
            }

            let overlap = overlapWordCount(
                previous: mergedNormalizedWords,
                current: normalizedWords,
                maxOverlapWords: maxOverlapWords
            )

            mergedDisplayWords.append(contentsOf: displayWords.dropFirst(overlap))
            mergedNormalizedWords.append(contentsOf: normalizedWords.dropFirst(overlap))
        }

        return cleanSegmentText(mergedDisplayWords.joined(separator: " "))
    }

    private static func overlapWordCount(previous: [String], current: [String], maxOverlapWords: Int) -> Int {
        let maxCount = min(maxOverlapWords, previous.count, current.count)
        guard maxCount >= 2 else {
            return 0
        }

        for overlapLength in stride(from: maxCount, through: 2, by: -1) {
            let previousSuffix = previous.suffix(overlapLength)
            let currentPrefix = current.prefix(overlapLength)
            if Array(previousSuffix) == Array(currentPrefix) {
                return overlapLength
            }
        }

        return 0
    }

    private static func normalizeText(_ text: String, lowercase: Bool) -> String {
        let normalized = text.precomposedStringWithCanonicalMapping
        var cleanedScalars = String.UnicodeScalarView()

        for scalar in normalized.unicodeScalars {
            if let character = Character(scalar).unicodeScalars.first.map(Character.init), removedChars.contains(character) {
                continue
            }
            if CharacterSet.controlCharacters.contains(scalar), !CharacterSet.whitespacesAndNewlines.contains(scalar) {
                continue
            }
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                cleanedScalars.append(UnicodeScalar(32)!)
                continue
            }
            cleanedScalars.append(scalar)
        }

        let collapsed = String(String(cleanedScalars).split(whereSeparator: \.isWhitespace).joined(separator: " "))
        return lowercase ? collapsed.lowercased() : collapsed
    }
}

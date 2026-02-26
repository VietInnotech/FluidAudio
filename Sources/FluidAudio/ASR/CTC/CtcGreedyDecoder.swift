import Foundation

/// CTC greedy decoder: argmax → collapse consecutive duplicates → remove blanks.
///
/// Operates on log-probability frames `[[Float]]` where each inner array has
/// `vocabSize` entries.  Returns decoded tokens with per-token timestamps and
/// confidences derived from the frame-level log-probabilities.
public struct CtcGreedyDecoder: Sendable {

    /// Vocabulary mapping token-id → string piece.
    public let vocabulary: [Int: String]

    /// Token id reserved for the CTC blank symbol.
    public let blankId: Int

    /// Sentencepiece-style word-boundary marker that prefixes word-initial tokens.
    /// Parakeet BPE uses "▁" (U+2581).
    private let wordBoundaryMarker: Character = "\u{2581}"

    public init(vocabulary: [Int: String], blankId: Int) {
        self.vocabulary = vocabulary
        self.blankId = blankId
    }

    // MARK: - Decoding

    /// Decode CTC log-probabilities into text with optional token timings.
    ///
    /// Algorithm:
    /// 1. Per frame: argmax over vocabulary → best token id
    /// 2. Collapse consecutive identical token ids
    /// 3. Remove blank tokens
    /// 4. Map remaining token ids to string pieces via vocabulary
    ///
    /// - Parameters:
    ///   - logProbs: Array of per-frame log-probability vectors `[T][V]`.
    ///   - frameDuration: Duration in seconds of each encoder frame.
    /// - Returns: `CtcDecodingResult` with text, token timings, and average confidence.
    public func decode(logProbs: [[Float]], frameDuration: Double) -> CtcDecodingResult {
        guard !logProbs.isEmpty else {
            return CtcDecodingResult(text: "", tokenTimings: [], confidence: 0)
        }

        // Step 1: Argmax per frame
        var frameTokens: [(tokenId: Int, logProb: Float, frameIndex: Int)] = []
        frameTokens.reserveCapacity(logProbs.count)
        for (frameIdx, frame) in logProbs.enumerated() {
            guard !frame.isEmpty else { continue }
            var bestId = 0
            var bestLogProb = frame[0]
            for v in 1..<frame.count {
                if frame[v] > bestLogProb {
                    bestLogProb = frame[v]
                    bestId = v
                }
            }
            frameTokens.append((tokenId: bestId, logProb: bestLogProb, frameIndex: frameIdx))
        }

        // Step 2 + 3: Collapse consecutive duplicates and remove blanks
        var collapsed: [(tokenId: Int, logProbSum: Float, logProbCount: Int, startFrame: Int, endFrame: Int)] = []
        var prevTokenId = -1
        for ft in frameTokens {
            if ft.tokenId == prevTokenId {
                // Same token as previous frame — extend the span
                collapsed[collapsed.count - 1].logProbSum += ft.logProb
                collapsed[collapsed.count - 1].logProbCount += 1
                collapsed[collapsed.count - 1].endFrame = ft.frameIndex
            } else {
                // New token
                if ft.tokenId != blankId {
                    collapsed.append(
                        (
                            tokenId: ft.tokenId,
                            logProbSum: ft.logProb,
                            logProbCount: 1,
                            startFrame: ft.frameIndex,
                            endFrame: ft.frameIndex
                        ))
                }
                prevTokenId = ft.tokenId
            }
        }

        // Step 4: Build token timings and assemble text
        var tokenTimings: [TokenTiming] = []
        tokenTimings.reserveCapacity(collapsed.count)
        var confidenceSum: Float = 0

        for entry in collapsed {
            let piece = vocabulary[entry.tokenId] ?? "<unk>"
            let avgLogProb = entry.logProbSum / Float(entry.logProbCount)
            let confidence = expf(avgLogProb)  // Convert log-prob to probability
            confidenceSum += confidence

            let startTime = Double(entry.startFrame) * frameDuration
            let endTime = Double(entry.endFrame + 1) * frameDuration

            tokenTimings.append(
                TokenTiming(
                    token: piece,
                    tokenId: entry.tokenId,
                    startTime: startTime,
                    endTime: endTime,
                    confidence: confidence
                ))
        }

        let text = assembleText(from: tokenTimings)
        let avgConfidence = collapsed.isEmpty ? 0 : confidenceSum / Float(collapsed.count)

        return CtcDecodingResult(
            text: text,
            tokenTimings: tokenTimings,
            confidence: avgConfidence
        )
    }

    // MARK: - Text Assembly

    /// Assemble final text from BPE token pieces.
    ///
    /// Parakeet BPE vocabulary uses "▁" (U+2581) as a word-boundary marker
    /// at the start of tokens that begin a new word.  This method:
    /// 1. Replaces leading "▁" with a space
    /// 2. Concatenates all pieces
    /// 3. Trims leading/trailing whitespace
    private func assembleText(from timings: [TokenTiming]) -> String {
        var result = ""
        result.reserveCapacity(timings.count * 4)

        for timing in timings {
            var piece = timing.token
            if piece.first == wordBoundaryMarker {
                piece = " " + String(piece.dropFirst())
            }
            result += piece
        }

        return result.trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Result Type

/// Result of CTC greedy decoding.
public struct CtcDecodingResult: Sendable {
    /// Decoded transcript text.
    public let text: String

    /// Per-token timing and confidence information.
    public let tokenTimings: [TokenTiming]

    /// Average token-level confidence (probability, 0–1).
    public let confidence: Float
}

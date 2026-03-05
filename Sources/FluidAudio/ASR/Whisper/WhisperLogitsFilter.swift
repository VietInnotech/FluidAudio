import CoreML
import Foundation

// Adapted from WhisperKit/Core/Text/LogitsFilter.swift
// MIT License, Copyright © 2024 Argmax, Inc.

// MARK: - Protocol

protocol WhisperLogitsFiltering {
    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray
}

// MARK: - Suppress Tokens Filter

/// Suppresses a fixed set of token IDs by setting their logit to -infinity.
final class WhisperSuppressTokensFilter: WhisperLogitsFiltering {
    private let suppressTokenIndexes: [[NSNumber]]

    init(suppressTokens: [Int]) {
        self.suppressTokenIndexes = suppressTokens.map { [0, 0, $0 as NSNumber] }
    }

    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        logits.fill(indexes: suppressTokenIndexes, with: -Float16.infinity)
        return logits
    }
}

// MARK: - Suppress Blank Filter

/// Suppresses the whitespace and EOT tokens at the very first decoded position.
final class WhisperSuppressBlankFilter: WhisperLogitsFiltering {
    private let sampleBegin: Int
    private let suppressTokenIndexes: [[NSNumber]]

    init(sampleBegin: Int) {
        self.sampleBegin = sampleBegin
        self.suppressTokenIndexes = [
            [0, 0, WhisperConfig.Tokens.whitespace as NSNumber],
            [0, 0, WhisperConfig.Tokens.endOfText as NSNumber],
        ]
    }

    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard tokens.count == sampleBegin else { return logits }
        logits.fill(indexes: suppressTokenIndexes, with: -Float16.infinity)
        return logits
    }
}

// MARK: - Timestamp Rules Filter

/// Enforces Whisper's timestamp pairing rules and forces a timestamp when
/// the sum of timestamp log-probabilities exceeds the max text log-probability.
///
/// Adapted from https://github.com/openai/whisper/blob/master/whisper/decoding.py#L441
final class WhisperTimestampRulesFilter: WhisperLogitsFiltering {
    private let sampleBegin: Int
    private let timeTokenBegin: Int
    private let endOfTextToken: Int
    private let noTimestampsIndex: [[NSNumber]]

    init(sampleBegin: Int) {
        self.sampleBegin = sampleBegin
        self.timeTokenBegin = WhisperConfig.Tokens.timeTokenBegin
        self.endOfTextToken = WhisperConfig.Tokens.endOfText
        self.noTimestampsIndex = [[0, 0, WhisperConfig.Tokens.noTimestamps as NSNumber]]
    }

    func filterLogits(_ logits: MLMultiArray, withTokens tokens: [Int]) -> MLMultiArray {
        guard sampleBegin <= tokens.count else { return logits }

        // Suppress <|notimestamps|>
        logits.fill(indexes: noTimestampsIndex, with: -Float16.infinity)

        if tokens.count > sampleBegin {
            let sampledTokens = tokens[sampleBegin...]
            let lastWasTimestamp = sampledTokens.last.map { $0 >= timeTokenBegin } ?? false
            let penultimateWasTimestamp =
                sampledTokens.count < 2
                || (sampledTokens.dropLast().last.map { $0 >= timeTokenBegin } ?? false)

            if lastWasTimestamp {
                if penultimateWasTimestamp {
                    // Two consecutive timestamps → force non-timestamp
                    logits.fillLastDimension(indexes: timeTokenBegin..<logits.count, with: -Float16.infinity)
                } else {
                    // Single timestamp → must be followed by timestamp or EOT
                    logits.fillLastDimension(indexes: 0..<endOfTextToken, with: -Float16.infinity)
                }
            }

            // Timestamps must not decrease
            let timestamps = sampledTokens.filter { $0 >= timeTokenBegin }
            if let lastTimestamp = timestamps.last {
                let timestampLast =
                    lastWasTimestamp && !penultimateWasTimestamp
                    ? lastTimestamp
                    : lastTimestamp + 1
                logits.fillLastDimension(indexes: timeTokenBegin..<timestampLast, with: -Float16.infinity)
            }
        }

        // If log-sum-exp of timestamp logits > max text logit, force a timestamp
        if sumOfTimestampProbabilityExceedsText(logits: logits) {
            logits.fillLastDimension(indexes: 0..<timeTokenBegin, with: -Float16.infinity)
        }

        return logits
    }

    /// Compares log-sum-exp of timestamp logits against the max text logit.
    /// Functionally identical to WhisperKit's BNNS-based implementation, using Float32 math.
    private func sumOfTimestampProbabilityExceedsText(logits: MLMultiArray) -> Bool {
        let totalCount = logits.count
        let timeBeginOffset = logits.linearOffset(for: [0, 0, timeTokenBegin as NSNumber])

        // Read all logits as Float32 (strides from withUnsafeBytes are in bytes)
        var logitsF32 = [Float](repeating: 0, count: totalCount)
        logits.withUnsafeBytes { ptr in
            let base = ptr.baseAddress!.bindMemory(to: Float16.self, capacity: totalCount)
            for i in 0..<totalCount { logitsF32[i] = Float(base[i]) }
        }

        // Numerically stable log-softmax: subtract global max before exp
        let globalMax = logitsF32.max() ?? 0
        let expSum = logitsF32.reduce(0.0) { $0 + Foundation.exp($1 - globalMax) }
        let logNorm = Foundation.log(expSum) + globalMax
        let logProbs = logitsF32.map { $0 - logNorm }

        // log-sum-exp of timestamp log-probs
        let timeLogProbs = logProbs[timeBeginOffset...]
        let maxTimeLP = timeLogProbs.max() ?? -Float.infinity
        let timeLogSumExp = Foundation.log(timeLogProbs.reduce(0.0) { $0 + Foundation.exp($1 - maxTimeLP) }) + maxTimeLP

        // max of text log-probs
        let maxTextLP = logProbs[0..<timeBeginOffset].max() ?? -Float.infinity

        return timeLogSumExp > maxTextLP
    }
}

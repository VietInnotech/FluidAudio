#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Benchmark for Whisper Large v3 Turbo on LibriSpeech.
///
/// Runs inference through `WhisperManager` with WER evaluation.
enum WhisperBenchmark {
    private static let logger = AppLogger(category: "WhisperBenchmark")

    static func runCLI(arguments: [String]) async {
        var subset = "test-clean"
        var maxFiles: Int? = nil
        var modelDir: String? = nil
        var outputFile = "whisper_benchmark_results.json"
        var language = "en"

        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--subset":
                if i + 1 < arguments.count {
                    subset = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--language", "-l":
                if i + 1 < arguments.count {
                    language = arguments[i + 1]
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        logger.info("Whisper Large v3 Turbo Benchmark")
        logger.info("  Subset: \(subset)")
        logger.info("  Max files: \(maxFiles?.description ?? "all")")
        logger.info("  Model dir: \(modelDir ?? "auto (local Models/)")")
        logger.info("  Language: \(language)")
        logger.info("  Output: \(outputFile)")

        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Whisper requires macOS 14 or later")
            exit(1)
        }

        do {
            // 1. Load Whisper models
            let manager = WhisperManager()
            let dir: URL
            if let modelDir {
                dir = URL(fileURLWithPath: modelDir)
            } else {
                // Default: Models/whisperkit-coreml/openai_whisper-large-v3_turbo/
                let projectRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                dir = projectRoot.appendingPathComponent(
                    "Models/whisperkit-coreml/openai_whisper-large-v3_turbo")
            }

            logger.info("Loading models from \(dir.path)")
            try await manager.loadModels(from: dir)

            // 2. Collect LibriSpeech files
            let libriDir = getLibriSpeechDirectory().appendingPathComponent(subset)
            guard FileManager.default.fileExists(atPath: libriDir.path) else {
                logger.error(
                    "LibriSpeech \(subset) not found at \(libriDir.path). Run 'fluidaudio download --dataset librispeech-\(subset)' first."
                )
                exit(1)
            }

            var files = collectLibriSpeechFiles(from: libriDir)
            if let maxFiles {
                files = Array(files.prefix(maxFiles))
            }

            logger.info("Found \(files.count) files in \(subset)")

            // 3. Run benchmark
            let converter = AudioConverter()
            var results: [WhisperBenchmarkFileResult] = []
            var totalWer: Float = 0
            var totalRtfx: Float = 0

            for (idx, file) in files.enumerated() {
                let audioSamples = try converter.resampleAudioFile(file.audioPath)
                let audioLength = Float(audioSamples.count) / 16000.0

                let start = CFAbsoluteTimeGetCurrent()
                let transcription = try await manager.transcribe(
                    audioSamples: audioSamples,
                    language: language
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                let rtfx = Float(audioLength) / Float(elapsed)

                // Compute WER (normalize: lowercase, strip punctuation)
                let wer = computeWER(
                    reference: normalizeForWER(file.transcript),
                    hypothesis: normalizeForWER(transcription)
                )

                totalWer += wer
                totalRtfx += rtfx

                let result = WhisperBenchmarkFileResult(
                    fileName: file.fileName,
                    reference: file.transcript,
                    hypothesis: transcription,
                    wer: wer,
                    audioLength: audioLength,
                    processingTime: Float(elapsed),
                    rtfx: rtfx
                )
                results.append(result)

                logger.info(
                    "[\(idx + 1)/\(files.count)] \(file.fileName) | WER: \(String(format: "%.1f", wer * 100))% | RTFx: \(String(format: "%.1f", rtfx))x | \(String(format: "%.1f", audioLength))s"
                )
                if wer > 0.1 {
                    logger.info("  REF: \(file.transcript)")
                    logger.info("  HYP: \(transcription)")
                }
            }

            // 4. Summary
            let avgWer = files.isEmpty ? 0 : totalWer / Float(files.count)
            let avgRtfx = files.isEmpty ? 0 : totalRtfx / Float(files.count)
            let medianRtfx = files.isEmpty ? 0 : median(results.map { $0.rtfx })

            logger.info("")
            logger.info("=== Whisper Large v3 Turbo Benchmark Results ===")
            logger.info("Files: \(files.count)")
            logger.info("Average WER: \(String(format: "%.1f", avgWer * 100))%")
            logger.info("Average RTFx: \(String(format: "%.1f", avgRtfx))x")
            logger.info("Median RTFx: \(String(format: "%.1f", medianRtfx))x")

            // 5. Save results
            let summary = WhisperBenchmarkSummary(
                model: "whisper-large-v3-turbo",
                dataset: "librispeech-\(subset)",
                language: language,
                fileCount: files.count,
                averageWer: avgWer,
                averageRtfx: avgRtfx,
                medianRtfx: medianRtfx,
                results: results
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(summary)
            try data.write(to: URL(fileURLWithPath: outputFile))
            logger.info("Results saved to \(outputFile)")

        } catch {
            logger.error("Benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - LibriSpeech Data Loading

    private static func getLibriSpeechDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDir = appSupport.appendingPathComponent("FluidAudio", isDirectory: true)
        return appDir.appendingPathComponent("Datasets/LibriSpeech", isDirectory: true)
    }

    private static func collectLibriSpeechFiles(from directory: URL) -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []

        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return files }

        var transcripts: [URL] = []
        while let url = enumerator.nextObject() as? URL {
            if url.lastPathComponent.hasSuffix(".trans.txt") {
                transcripts.append(url)
            }
        }

        for transFile in transcripts {
            guard let contents = try? String(contentsOf: transFile, encoding: .utf8) else { continue }
            let parentDir = transFile.deletingLastPathComponent()

            for line in contents.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { continue }

                let parts = trimmed.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }

                let audioId = String(parts[0])
                let transcript = String(parts[1])
                let audioPath = parentDir.appendingPathComponent("\(audioId).flac")

                guard FileManager.default.fileExists(atPath: audioPath.path) else { continue }

                files.append(LibriSpeechFile(
                    fileName: audioId,
                    audioPath: audioPath,
                    transcript: transcript
                ))
            }
        }

        return files.sorted { $0.fileName < $1.fileName }
    }

    // MARK: - WER Computation

    /// Normalize text for WER: lowercase, strip punctuation, collapse whitespace.
    private static func normalizeForWER(_ text: String) -> String {
        let lowered = text.lowercased()
        // Remove all characters that aren't letters, numbers, or whitespace
        let stripped = lowered.unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0) || CharacterSet.whitespaces.contains($0)
        }
        let cleaned = String(String.UnicodeScalarView(stripped))
        // Collapse multiple spaces
        return cleaned.split(separator: " ").joined(separator: " ")
    }

    private static func computeWER(reference: String, hypothesis: String) -> Float {
        let refWords = reference.split(separator: " ").map { String($0) }
        let hypWords = hypothesis.split(separator: " ").map { String($0) }

        guard !refWords.isEmpty else { return hypWords.isEmpty ? 0 : 1 }

        let m = refWords.count
        let n = hypWords.count

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if refWords[i - 1] == hypWords[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
                }
            }
        }

        return Float(dp[m][n]) / Float(m)
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 0 {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }

    // MARK: - Usage

    private static func printUsage() {
        print("""
            Usage: fluidaudio whisper-benchmark [OPTIONS]

            Options:
              --subset <name>        LibriSpeech subset (default: test-clean)
              --max-files <n>        Maximum number of files to process
              --model-dir <path>     Path to Whisper model directory
              --language <code>      Language code (default: en)
              --output <file>        Output JSON file (default: whisper_benchmark_results.json)
              -h, --help             Show this help message
            """)
    }
}

// MARK: - Result Types

private struct WhisperBenchmarkFileResult: Codable {
    let fileName: String
    let reference: String
    let hypothesis: String
    let wer: Float
    let audioLength: Float
    let processingTime: Float
    let rtfx: Float
}

private struct WhisperBenchmarkSummary: Codable {
    let model: String
    let dataset: String
    let language: String
    let fileCount: Int
    let averageWer: Float
    let averageRtfx: Float
    let medianRtfx: Float
    let results: [WhisperBenchmarkFileResult]
}
#endif

#if os(macOS)
import FluidAudio
import Foundation

/// Command to transcribe audio files using CTC greedy decoding.
///
/// Uses `CtcAsrManager` with the Vietnamese-finetuned Parakeet CTC 0.6B model.
///
/// Usage:
///   fluidaudio ctc-transcribe audio.wav
///   fluidaudio ctc-transcribe audio.wav --model-dir ./Models/parakeet-ctc-0.6b-vietnamese-coreml
enum CtcTranscribeCommand {
    private static let logger = AppLogger(category: "CtcTranscribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var modelDir: String?

        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                } else {
                    logger.error("--model-dir requires a path argument")
                    exit(1)
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        await transcribe(audioFile: audioFile, modelDir: modelDir)
    }

    // MARK: - Private

    private static func transcribe(audioFile: String, modelDir: String?) async {
        let manager = CtcAsrManager()

        do {
            if let dir = modelDir {
                logger.info("Loading CTC models from: \(dir)")
                let dirURL = URL(fileURLWithPath: dir)
                try await manager.loadModels(from: dirURL, variant: .ctcVietnamese)
            } else {
                logger.info("Downloading CTC Vietnamese models from HuggingFace...")
                try await manager.loadModels(variant: .ctcVietnamese)
            }
        } catch {
            logger.error("Failed to load models: \(error.localizedDescription)")
            exit(1)
        }

        do {
            logger.info("Transcribing: \(audioFile)")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await manager.transcribe(url: URL(fileURLWithPath: audioFile))
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            logger.info(String(repeating: "=", count: 60))
            logger.info("CTC TRANSCRIPTION")
            logger.info(String(repeating: "=", count: 60))
            print(result.text)
            logger.info("")
            logger.info("Performance:")
            logger.info("  Audio duration:  \(String(format: "%.2f", result.duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", elapsed))s")
            logger.info("  RTFx:            \(String(format: "%.2f", result.duration / elapsed))x")
            logger.info("  Confidence:      \(String(format: "%.3f", result.confidence))")
        } catch {
            logger.error("Transcription failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudio ctc-transcribe <audio_file> [options]

            Transcribe an audio file using CTC greedy decoding with the Parakeet CTC Vietnamese model.

            Arguments:
              <audio_file>     Path to audio file (WAV, MP3, FLAC, M4A, etc.)

            Options:
              --model-dir <path>   Path to directory with compiled .mlmodelc models and vocab.json.
                                   If omitted, models are auto-downloaded from HuggingFace.
              --help, -h           Show this help message

            Examples:
              fluidaudio ctc-transcribe audio.wav
              fluidaudio ctc-transcribe audio.wav --model-dir ./Models/parakeet-ctc-0.6b-vietnamese-coreml
            """
        )
    }
}
#endif

#if os(macOS)
import FluidAudio
import Foundation

/// Command to transcribe audio files using Whisper Large v3 Turbo.
enum WhisperTranscribeCommand {
    private static let logger = AppLogger(category: "WhisperTranscribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var modelDir: String?
        var language: String = "en"
        var variant: WhisperModelVariant = .standard

        // Parse options
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
                }
            case "--language", "-l":
                if i + 1 < arguments.count {
                    language = arguments[i + 1]
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "standard":
                        variant = .standard
                    case "erax", "erax-wow-turbo":
                        variant = .eraXWowTurbo
                    default:
                        logger.error("Invalid variant: \(arguments[i + 1]). Use 'standard' or 'erax'")
                        exit(1)
                    }
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        await transcribe(audioFile: audioFile, modelDir: modelDir, language: language, variant: variant)
    }

    private static func transcribe(
        audioFile: String,
        modelDir: String?,
        language: String,
        variant: WhisperModelVariant
    ) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Whisper requires macOS 14 or later")
            return
        }

        do {
            let manager = WhisperManager()
            let variantName = variant == .standard ? "Standard Whisper" : "EraX-WoW-Turbo"

            if let dir = modelDir {
                logger.info("Loading Whisper models from: \(dir)")
                try await manager.loadModels(from: URL(fileURLWithPath: dir))
            } else {
                logger.info("Downloading \(variantName) models from HuggingFace...")
                let cacheDir = try await WhisperModels.download(variant: variant)
                try await manager.loadModels(from: cacheDir)
            }

            // Load and resample audio to 16kHz mono
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(samples.count) / Double(WhisperConfig.sampleRate)
            logger.info(
                "Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 16kHz"
            )

            // Transcribe
            logger.info("Transcribing with \(variantName) (language: \(language))...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let text = try await manager.transcribe(audioSamples: samples, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtfx = duration / elapsed

            // Output
            logger.info(String(repeating: "=", count: 50))
            logger.info("WHISPER TRANSCRIPTION (\(variantName))")
            logger.info(String(repeating: "=", count: 50))
            print(text)
            logger.info("")
            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", elapsed))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
        } catch {
            logger.error("Transcription failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        logger.info("""
            Usage: fluidaudio whisper-transcribe <audio-file> [options]

            Options:
              --model-dir <path>    Path to local Whisper model directory (auto-downloads if not specified)
              --variant <name>      Model variant: 'standard' (default) or 'erax' for EraX-WoW-Turbo
              --language, -l <code> Language code (e.g. en, vi, fr, de). Default: en
              --help, -h            Show this help message

            Examples:
              fluidaudio whisper-transcribe audio.wav
              fluidaudio whisper-transcribe audio.wav --variant erax --language vi
              fluidaudio whisper-transcribe audio.wav --language fr
              fluidaudio whisper-transcribe audio.wav --model-dir Models/whisperkit-coreml/openai_whisper-large-v3-v20240930
            """)
    }
}
#endif

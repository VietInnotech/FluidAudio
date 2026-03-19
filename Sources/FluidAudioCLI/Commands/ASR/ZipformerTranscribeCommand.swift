#if os(macOS)
import FluidAudio
import Foundation

enum ZipformerTranscribeCommand {
    private static let logger = AppLogger(category: "ZipformerTranscribe")

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
                guard i + 1 < arguments.count else {
                    logger.error("--model-dir requires a path argument")
                    exit(1)
                }
                modelDir = arguments[i + 1]
                i += 1
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        let manager = ZipformerVnAsrManager()

        do {
            if let modelDir {
                try await manager.loadModels(from: URL(fileURLWithPath: modelDir))
            } else {
                try await manager.loadModels()
            }

            let result = try await manager.transcribe(url: URL(fileURLWithPath: audioFile))
            print(result.text)
            logger.info("RTFx: \(String(format: "%.2f", result.rtfx))x")
        } catch {
            logger.error("Zipformer transcription failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudio zipformer-transcribe <audio_file> [options]

            Transcribe using Zipformer VN (sherpa-onnx model format) with native Swift inference.

            Options:
              --model-dir <path>   Path to model dir with encoder.int8.onnx, decoder.onnx, joiner.int8.onnx, tokens.txt
              --help, -h           Show help

            Examples:
              fluidaudio zipformer-transcribe audio.wav
              fluidaudio zipformer-transcribe audio.wav --model-dir /Users/vit/zipformer-macos/models/stt/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09
            """
        )
    }
}
#endif

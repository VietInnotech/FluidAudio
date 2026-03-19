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
        var streamingModelDir: String?
        var useVad = true
        var backend = ZipformerVnBackend.preferred

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
            case "--streaming-model-dir":
                guard i + 1 < arguments.count else {
                    logger.error("--streaming-model-dir requires a path argument")
                    exit(1)
                }
                streamingModelDir = arguments[i + 1]
                i += 1
            case "--backend":
                guard i + 1 < arguments.count else {
                    logger.error("--backend requires one of: native, sherpa-offline, sherpa-streaming")
                    exit(1)
                }
                guard let parsed = ZipformerVnBackend(rawValue: arguments[i + 1]) else {
                    logger.error("Invalid backend: \(arguments[i + 1])")
                    exit(1)
                }
                backend = parsed
                i += 1
            case "--no-vad":
                useVad = false
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

            let result = try await manager.transcribe(
                url: URL(fileURLWithPath: audioFile),
                useVad: useVad,
                backend: backend,
                streamingModelDirectory: streamingModelDir.map { URL(fileURLWithPath: $0) }
            )
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

            Transcribe using Zipformer VN with selectable native backends.

            Options:
              --model-dir <path>   Path to model dir with encoder.int8.onnx, decoder.onnx, joiner.int8.onnx, tokens.txt
              --backend <name>     Backend: native, sherpa-offline, sherpa-streaming
              --streaming-model-dir <path>
                                   Path to streaming RNNT model dir when using sherpa-streaming
              --no-vad             Disable VAD-first segmentation and decode the full file directly
              --help, -h           Show help

            Examples:
              fluidaudio zipformer-transcribe audio.wav
              fluidaudio zipformer-transcribe audio.wav --backend sherpa-offline
              fluidaudio zipformer-transcribe audio.wav --backend sherpa-streaming --streaming-model-dir /Users/vit/zipformer-macos/models/stt/hynt-Zipformer-30M-RNNT-Streaming-6000h
              fluidaudio zipformer-transcribe audio.wav --no-vad
              fluidaudio zipformer-transcribe audio.wav --model-dir /Users/vit/zipformer-macos/models/stt/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09
            """
        )
    }
}
#endif

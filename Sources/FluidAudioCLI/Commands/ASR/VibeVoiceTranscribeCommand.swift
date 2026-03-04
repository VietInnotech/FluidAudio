#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Command to transcribe audio files using VibeVoice-ASR (unified ASR + diarization + timestamps).
enum VibeVoiceTranscribeCommand {
    private static let logger = AppLogger(category: "VibeVoiceTranscribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        if arguments[0] == "--help" || arguments[0] == "-h" {
            printUsage()
            exit(0)
        }

        let audioFile = arguments[0]
        var modelDir: String?
        var context: String?
        var variant: VibeVoiceAsrVariant = .f32
        var maxTokens: Int = 8192
        var jsonOutput = false

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
            case "--context", "-c":
                if i + 1 < arguments.count {
                    context = arguments[i + 1]
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    let v = arguments[i + 1].lowercased()
                    if let parsed = VibeVoiceAsrVariant(rawValue: v) {
                        variant = parsed
                    } else {
                        logger.error("Unknown variant '\(arguments[i + 1])'. Use 'f32' or 'int4'.")
                        exit(1)
                    }
                    i += 1
                }
            case "--max-tokens":
                if i + 1 < arguments.count, let val = Int(arguments[i + 1]) {
                    maxTokens = val
                    i += 1
                }
            case "--json":
                jsonOutput = true
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        await transcribe(
            audioFile: audioFile,
            modelDir: modelDir,
            context: context,
            variant: variant,
            maxTokens: maxTokens,
            jsonOutput: jsonOutput
        )
    }

    private static func transcribe(
        audioFile: String,
        modelDir: String?,
        context: String?,
        variant: VibeVoiceAsrVariant,
        maxTokens: Int,
        jsonOutput: Bool
    ) async {
        guard #available(macOS 15, iOS 18, *) else {
            logger.error("VibeVoice-ASR requires macOS 15 or later")
            return
        }

        do {
            // Load models
            let manager = VibeVoiceAsrManager()

            if let dir = modelDir {
                logger.info("Loading VibeVoice-ASR models from: \(dir)")
                let dirURL = URL(fileURLWithPath: dir)
                try await manager.loadModels(from: dirURL)
            } else {
                logger.info(
                    "Downloading VibeVoice-ASR \(variant.rawValue) models from HuggingFace..."
                )
                let cacheDir = try await VibeVoiceAsrModels.download(variant: variant)
                try await manager.loadModels(from: cacheDir)
            }

            // Load and resample audio to 24kHz mono
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(VibeVoiceAsrConfig.sampleRate),
                channels: 1,
                interleaved: false
            )!
            let samples = try AudioConverter(targetFormat: targetFormat).resampleAudioFile(
                path: audioFile
            )
            let duration = Double(samples.count) / Double(VibeVoiceAsrConfig.sampleRate)
            logger.info(
                "Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 24kHz"
            )

            // Transcribe
            logger.info(
                "Transcribing with VibeVoice-ASR (context: \(context ?? "none"), maxTokens: \(maxTokens))..."
            )
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await manager.transcribe(
                audioSamples: samples,
                context: context,
                maxNewTokens: maxTokens
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtfx = duration / elapsed

            if jsonOutput {
                // JSON output mode
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let jsonData = try encoder.encode(result.segments)
                if let jsonString = String(data: jsonData, encoding: .utf8) {
                    print(jsonString)
                }
            } else {
                // Human-readable output
                logger.info(String(repeating: "=", count: 60))
                logger.info("VIBEVOICE-ASR TRANSCRIPTION")
                logger.info(String(repeating: "=", count: 60))

                if result.segments.isEmpty {
                    logger.info("No segments parsed from model output.")
                    logger.info("Raw output:")
                    print(result.rawText)
                } else {
                    for segment in result.segments {
                        let line =
                            "[\(segment.startTime) - \(segment.endTime)] \(segment.speakerId): \(segment.content)"
                        print(line)
                    }

                    logger.info("")
                    logger.info("Summary:")
                    logger.info("  Segments: \(result.segments.count)")
                    logger.info("  Speakers: \(result.speakerCount)")
                    if let dur = result.totalDuration {
                        logger.info(
                            "  Audio covered: \(String(format: "%.2f", dur))s"
                        )
                    }
                }

                logger.info("")
                logger.info("Performance:")
                logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
                logger.info("  Processing time: \(String(format: "%.2f", elapsed))s")
                logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")
            }

        } catch {
            logger.error("VibeVoice-ASR transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        logger.info(
            """

            VibeVoice-ASR Transcribe Command

            Unified ASR + speaker diarization + timestamping in a single pass.
            Supports audio up to 60 minutes. Outputs structured segments with
            speaker IDs and timestamps.

            Usage: fluidaudio vibevoice-transcribe <audio_file> [options]

            Options:
                --help, -h              Show this help message
                --model-dir <path>      Path to local model directory (skips download)
                --variant <f32|int4>    Model variant (default: f32).
                                        int4 uses ~3.5 GB (vs ~14 GB for f32).
                --context, -c <text>    Context to improve accuracy (e.g., speaker names,
                                        domain terms, technical vocabulary)
                --max-tokens <n>        Maximum tokens to generate (default: 8192)
                --json                  Output raw JSON array of segments

            Examples:
                fluidaudio vibevoice-transcribe meeting.wav
                fluidaudio vibevoice-transcribe meeting.wav --variant int4
                fluidaudio vibevoice-transcribe interview.mp3 --context "Interview with John Smith about AI"
                fluidaudio vibevoice-transcribe call.wav --json > segments.json
            """
        )
    }
}
#endif

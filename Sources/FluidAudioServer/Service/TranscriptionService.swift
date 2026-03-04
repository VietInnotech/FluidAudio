import FluidAudio
import Foundation
import Logging

// Swift-log logger for terminal-visible output
private let serviceLogger = Logger(label: "FluidAudioServer.TranscriptionService")

/// Type-erased wrapper for Qwen3AsrManager to avoid availability annotations leaking to the actor.
///
/// Qwen3AsrManager/Qwen3AsrModels require macOS 15+. We wrap the manager in an `Any` box
/// and only touch it behind `#available` checks.
@available(macOS 15, *)
private struct Qwen3Backend {
    let manager: Qwen3AsrManager

    init() {
        self.manager = Qwen3AsrManager()
    }

    func loadModels(from directory: URL) async throws {
        try await manager.loadModels(from: directory)
    }

    func transcribe(audioSamples: [Float], language: String?) async throws -> String {
        try await manager.transcribe(audioSamples: audioSamples, language: language, maxNewTokens: 512)
    }

    /// Transcribe long audio by VAD-segmenting into chunks that fit the Qwen3 context window,
    /// then concatenating results. Follows the same pattern as Qwen3-ASR-Toolkit and antirez/qwen-asr.
    func transcribeWithVad(
        audioSamples: [Float],
        vadManager: VadManager,
        language: String?
    ) async throws -> String {
        let segments = try await vadManager.segmentSpeech(
            audioSamples,
            config: .qwen3Optimized
        )

        guard !segments.isEmpty else {
            return ""
        }

        let sampleRate = 16_000
        var transcribedParts: [String] = []

        for (idx, segment) in segments.enumerated() {
            let startSample = max(0, Int(segment.startTime * Double(sampleRate)))
            let endSample = min(audioSamples.count, Int(segment.endTime * Double(sampleRate)))

            guard endSample > startSample else { continue }

            let segmentSamples = Array(audioSamples[startSample..<endSample])
            // Skip segments shorter than 0.5s — too short for meaningful transcription
            guard segmentSamples.count >= sampleRate / 2 else { continue }

            let segDuration = Double(segmentSamples.count) / Double(sampleRate)
            serviceLogger.debug(
                "qwen3 chunk \(idx + 1)/\(segments.count): \(String(format: "%.1f", segment.startTime))-\(String(format: "%.1f", segment.endTime))s (\(String(format: "%.1f", segDuration))s)"
            )

            let text = try await manager.transcribe(
                audioSamples: segmentSamples,
                language: language,
                maxNewTokens: 512
            )

            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                transcribedParts.append(trimmed)
            }
        }

        return transcribedParts.joined(separator: " ")
    }
}

/// Thread-safe transcription service that allows one active transcription at a time.
///
/// Manages model lifecycle: keeps only one backend loaded in memory.
/// On model switch, the old manager is cleaned up before loading the new one.
actor TranscriptionService {

    private let config: ServerConfig
    private let audioConverter = AudioConverter()

    // Currently loaded backend state
    private var currentModel: ModelID?
    // AsrManager is a class (not Sendable). We guarantee single-access via the actor.
    nonisolated(unsafe) private var parakeetManager: AsrManager?
    private var parakeetVadManager: VadManager?
    // Stored as Any to avoid propagating @available requirement on the actor
    private var qwen3Backend: Any?
    private var qwen3VadManager: VadManager?
    private var ctcManager: CtcAsrManager?
    // WhisperManager is an actor (Sendable), stored directly.
    private var whisperManager: WhisperManager?

    // Busy flag for single-concurrency enforcement
    private var isBusy = false

    init(config: ServerConfig) {
        self.config = config
    }

    // MARK: - Public API

    /// Transcribe audio samples using the specified model.
    ///
    /// Returns 429 (via error) if another transcription is already in progress.
    func transcribe(
        audioSamples: [Float],
        model: ModelID,
        language: String?
    ) async throws -> TranscriptionResult {
        guard !isBusy else {
            serviceLogger.warning("rejected: server busy (model=\(model.rawValue))")
            throw ServerError.tooManyRequests("Server is currently processing another request. Try again later.")
        }

        // Validate audio duration
        let durationSeconds = Double(audioSamples.count) / 16_000.0
        guard durationSeconds <= Double(config.maxAudioSeconds) else {
            throw ServerError.badRequest(
                "Audio duration \(String(format: "%.1f", durationSeconds))s exceeds maximum of \(config.maxAudioSeconds)s"
            )
        }

        isBusy = true
        defer { isBusy = false }

        // Switch model if needed
        try await ensureModelLoaded(model)

        return try await performTranscription(audioSamples: audioSamples, model: model, language: language)
    }

    // MARK: - Streaming API

    /// Acquire the service for a streaming session. Sets the server as busy and loads the model.
    ///
    /// Call `releaseStreaming()` when the session ends to allow other requests.
    func acquireForStreaming(model: ModelID) async throws {
        guard !isBusy else {
            serviceLogger.warning("streaming rejected: server busy")
            throw ServerError.tooManyRequests("Server is currently processing another request. Try again later.")
        }
        isBusy = true
        do {
            try await ensureModelLoaded(model)
        } catch {
            isBusy = false
            throw error
        }
        serviceLogger.info("streaming session acquired: \(model.rawValue)")
    }

    /// Release the service after a streaming session ends.
    func releaseStreaming() {
        isBusy = false
        serviceLogger.info("streaming session released")
    }

    /// Transcribe audio samples during a streaming session (skips busy check).
    ///
    /// Must only be called between `acquireForStreaming()` and `releaseStreaming()`.
    func transcribeForStream(
        audioSamples: [Float],
        model: ModelID,
        language: String?
    ) async throws -> TranscriptionResult {
        return try await performTranscription(audioSamples: audioSamples, model: model, language: language)
    }

    // MARK: - Core Transcription

    /// Core transcription logic without concurrency guards.
    private func performTranscription(
        audioSamples: [Float],
        model: ModelID,
        language: String?
    ) async throws -> TranscriptionResult {
        let durationSeconds = Double(audioSamples.count) / 16_000.0
        let startTime = CFAbsoluteTimeGetCurrent()

        // Route transcription to the correct backend
        let text: String

        switch model.backend {
        case .parakeet:
            guard let manager = parakeetManager else {
                throw ServerError.internalError("Parakeet manager not initialized")
            }
            guard let vadManager = parakeetVadManager else {
                throw ServerError.internalError("Parakeet VAD manager not initialized")
            }
            let result = try await manager.transcribeWithVad(
                audioSamples,
                vadManager: vadManager,
                source: .system,
                segmentationConfig: .asrOptimized
            )
            text = result.text

        case .qwen3:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            guard let backend = qwen3Backend as? Qwen3Backend else {
                throw ServerError.internalError("Qwen3 backend not initialized")
            }
            guard let vadManager = qwen3VadManager else {
                throw ServerError.internalError("Qwen3 VAD manager not initialized")
            }
            // Use VAD-chunked transcription to handle audio of any length.
            // Qwen3's KV-cache is limited to 512 tokens (~25s audio).
            text = try await backend.transcribeWithVad(
                audioSamples: audioSamples,
                vadManager: vadManager,
                language: language
            )

        case .ctc:
            guard let manager = ctcManager else {
                throw ServerError.internalError("CTC manager not initialized")
            }
            let result = try await manager.transcribe(audioSamples: audioSamples)
            text = result.text

        case .whisper:
            guard let manager = whisperManager else {
                throw ServerError.internalError("Whisper manager not initialized")
            }
            text = try await manager.transcribe(audioSamples: audioSamples, language: language)
        }

        let processingTime = CFAbsoluteTimeGetCurrent() - startTime

        return TranscriptionResult(
            text: text,
            model: model.rawValue,
            duration: durationSeconds,
            processingTime: processingTime
        )
    }

    // MARK: - Model Lifecycle

    private func ensureModelLoaded(_ model: ModelID) async throws {
        guard model != currentModel else { return }

        serviceLogger.info(
            "switching model: \(self.currentModel?.rawValue ?? "none") → \(model.rawValue)"
        )

        // Cleanup previous backend
        cleanupCurrentBackend()

        let loadStart = CFAbsoluteTimeGetCurrent()

        // Load the requested model
        switch model {
        case .parakeetV2:
            serviceLogger.info("downloading parakeet-tdt-v2 models…")
            let models = try await AsrModels.downloadAndLoad(version: .v2)
            serviceLogger.info("initializing parakeet-tdt-v2…")
            let manager = AsrManager()
            try await manager.initialize(models: models)
            parakeetManager = manager
            serviceLogger.info("loading VAD…")
            parakeetVadManager = try await VadManager(config: .default)

        case .parakeetV3:
            serviceLogger.info("downloading parakeet-tdt-v3 models…")
            let models = try await AsrModels.downloadAndLoad(version: .v3)
            serviceLogger.info("initializing parakeet-tdt-v3…")
            let manager = AsrManager()
            try await manager.initialize(models: models)
            parakeetManager = manager
            serviceLogger.info("loading VAD…")
            parakeetVadManager = try await VadManager(config: .default)

        case .qwen3F32:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            serviceLogger.info("downloading qwen3-asr f32 models…")
            let backend = Qwen3Backend()
            _ = try await Qwen3AsrModels.download(variant: .f32)
            let cacheDir = Qwen3AsrModels.defaultCacheDirectory(variant: .f32)
            serviceLogger.info("initializing qwen3-asr f32…")
            try await backend.loadModels(from: cacheDir)
            qwen3Backend = backend
            serviceLogger.info("loading VAD for qwen3 chunking…")
            qwen3VadManager = try await VadManager(config: .default)

        case .qwen3Int8:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            serviceLogger.info("downloading qwen3-asr int8 models…")
            let backend = Qwen3Backend()
            _ = try await Qwen3AsrModels.download(variant: .int8)
            let cacheDir = Qwen3AsrModels.defaultCacheDirectory(variant: .int8)
            serviceLogger.info("initializing qwen3-asr int8…")
            try await backend.loadModels(from: cacheDir)
            qwen3Backend = backend
            serviceLogger.info("loading VAD for qwen3 chunking…")
            qwen3VadManager = try await VadManager(config: .default)

        case .qwen3Large:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            serviceLogger.info("downloading qwen3-asr 1.7b models…")
            let backend = Qwen3Backend()
            _ = try await Qwen3AsrModels.download(variant: .large)
            let cacheDir = Qwen3AsrModels.defaultCacheDirectory(variant: .large)
            serviceLogger.info("initializing qwen3-asr 1.7b…")
            try await backend.loadModels(from: cacheDir)
            qwen3Backend = backend
            serviceLogger.info("loading VAD for qwen3 chunking…")
            qwen3VadManager = try await VadManager(config: .default)

        case .ctcVi:
            serviceLogger.info("downloading CTC Vietnamese models…")
            let manager = CtcAsrManager()
            try await manager.loadModels(variant: .ctcVietnamese)
            ctcManager = manager

        case .whisper:
            serviceLogger.info("downloading Whisper Large v3 Turbo models…")
            let cacheDir = try await WhisperModels.download()
            serviceLogger.info("initializing Whisper…")
            let manager = WhisperManager()
            try await manager.loadModels(from: cacheDir)
            whisperManager = manager
        }

        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        currentModel = model
        serviceLogger.info("model ready: \(model.rawValue) (\(String(format: "%.1f", loadTime))s)")
    }

    private func cleanupCurrentBackend() {
        if let manager = parakeetManager {
            manager.cleanup()
            parakeetManager = nil
        }
        parakeetVadManager = nil
        qwen3Backend = nil
        qwen3VadManager = nil
        ctcManager = nil
        whisperManager = nil
        currentModel = nil
    }
}

// MARK: - Result Types

/// Internal result from a transcription operation.
struct TranscriptionResult: Sendable {
    let text: String
    let model: String
    let duration: Double
    let processingTime: Double
}

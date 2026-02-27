import FluidAudio
import Foundation
import OSLog

private let serviceLogger = Logger(subsystem: "FluidAudioServer", category: "TranscriptionService")

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
    // Stored as Any to avoid propagating @available requirement on the actor
    private var qwen3Backend: Any?
    private var ctcManager: CtcAsrManager?

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

        let startTime = CFAbsoluteTimeGetCurrent()

        // Route transcription to the correct backend
        let text: String
        let audioDuration = durationSeconds

        switch model.backend {
        case .parakeet:
            guard let manager = parakeetManager else {
                throw ServerError.internalError("Parakeet manager not initialized")
            }
            let result = try await manager.transcribe(audioSamples, source: .system)
            text = result.text

        case .qwen3:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            guard let backend = qwen3Backend as? Qwen3Backend else {
                throw ServerError.internalError("Qwen3 backend not initialized")
            }
            text = try await backend.transcribe(audioSamples: audioSamples, language: language)

        case .ctc:
            guard let manager = ctcManager else {
                throw ServerError.internalError("CTC manager not initialized")
            }
            let result = try await manager.transcribe(audioSamples: audioSamples)
            text = result.text
        }

        let processingTime = CFAbsoluteTimeGetCurrent() - startTime

        return TranscriptionResult(
            text: text,
            model: model.rawValue,
            duration: audioDuration,
            processingTime: processingTime
        )
    }

    // MARK: - Model Lifecycle

    private func ensureModelLoaded(_ model: ModelID) async throws {
        guard model != currentModel else { return }

        serviceLogger.info("Switching model from \(self.currentModel?.rawValue ?? "none") to \(model.rawValue)")

        // Cleanup previous backend
        cleanupCurrentBackend()

        // Load the requested model
        switch model {
        case .parakeetV2:
            let models = try await AsrModels.downloadAndLoad(version: .v2)
            let manager = AsrManager()
            try await manager.initialize(models: models)
            parakeetManager = manager

        case .parakeetV3:
            let models = try await AsrModels.downloadAndLoad(version: .v3)
            let manager = AsrManager()
            try await manager.initialize(models: models)
            parakeetManager = manager

        case .qwen3F32:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            let backend = Qwen3Backend()
            _ = try await Qwen3AsrModels.download(variant: .f32)
            let cacheDir = Qwen3AsrModels.defaultCacheDirectory(variant: .f32)
            try await backend.loadModels(from: cacheDir)
            qwen3Backend = backend

        case .qwen3Int8:
            guard #available(macOS 15, *) else {
                throw ServerError.internalError("Qwen3 models require macOS 15 or later")
            }
            let backend = Qwen3Backend()
            _ = try await Qwen3AsrModels.download(variant: .int8)
            let cacheDir = Qwen3AsrModels.defaultCacheDirectory(variant: .int8)
            try await backend.loadModels(from: cacheDir)
            qwen3Backend = backend

        case .ctcVi:
            let manager = CtcAsrManager()
            try await manager.loadModels(variant: .ctcVietnamese)
            ctcManager = manager
        }

        currentModel = model
        serviceLogger.info("Model \(model.rawValue) loaded successfully")
    }

    private func cleanupCurrentBackend() {
        if let manager = parakeetManager {
            manager.cleanup()
            parakeetManager = nil
        }
        qwen3Backend = nil
        ctcManager = nil
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

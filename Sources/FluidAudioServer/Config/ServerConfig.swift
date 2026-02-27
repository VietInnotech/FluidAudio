import Foundation

/// Configuration for the FluidAudio ASR server, loaded from environment variables.
struct ServerConfig: Sendable {
    let host: String
    let port: Int
    let apiKey: String?
    let maxUploadBytes: Int
    let maxAudioSeconds: Int

    /// Default configuration from environment variables.
    static func fromEnvironment() -> ServerConfig {
        let host = ProcessInfo.processInfo.environment["FLUIDAUDIO_SERVER_HOST"] ?? "127.0.0.1"
        let port = Int(ProcessInfo.processInfo.environment["FLUIDAUDIO_SERVER_PORT"] ?? "") ?? 8080
        let apiKey = ProcessInfo.processInfo.environment["FLUIDAUDIO_SERVER_API_KEY"]
        let maxUploadMB = Int(ProcessInfo.processInfo.environment["FLUIDAUDIO_SERVER_MAX_UPLOAD_MB"] ?? "") ?? 128
        let maxAudioSeconds =
            Int(ProcessInfo.processInfo.environment["FLUIDAUDIO_SERVER_MAX_AUDIO_SECONDS"] ?? "") ?? 900

        return ServerConfig(
            host: host,
            port: port,
            apiKey: apiKey,
            maxUploadBytes: maxUploadMB * 1024 * 1024,
            maxAudioSeconds: maxAudioSeconds
        )
    }
}

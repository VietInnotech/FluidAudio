import Foundation
import Hummingbird
import Logging
import NIOCore

// Load .env file if it exists
loadDotEnv()

// Configure logging system globally
let logLevelEnv = ProcessInfo.processInfo.environment["FLUIDAUDIO_LOG_LEVEL"] ?? "info"
let globalLogLevel = Logger.Level(rawValue: logLevelEnv.lowercased()) ?? .info
LoggingSystem.bootstrap { label in
    var handler = StreamLogHandler.standardError(label: label)
    handler.logLevel = globalLogLevel
    return handler
}

let config = ServerConfig.fromEnvironment()

var logger = Logger(label: "FluidAudioServer")
logger.info("Starting FluidAudio ASR Server")
logger.info("Host: \(config.host)")
logger.info("Port: \(config.port)")
logger.info("Max upload: \(config.maxUploadBytes / 1024 / 1024) MB")
logger.info("Max audio duration: \(config.maxAudioSeconds)s")
logger.info("API key: \(config.apiKey != nil ? "configured" : "disabled (open access)")")

let service = TranscriptionService(config: config)

let router = Router()

// Request/response access log (outermost middleware — times the full request lifecycle)
router.add(middleware: RequestLoggingMiddleware())

// Health check — always unauthenticated, registered before any auth middleware
router.get("health") { _, _ -> Response in
    let body = #"{"status":"ok"}"#
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(string: body))
    )
}

// Apply auth middleware globally if API key is configured
if let apiKey = config.apiKey {
    router.add(middleware: AuthMiddleware(apiKey: apiKey))
}

// Register API routes (auth middleware is already on the router)
registerRoutes(on: router, service: service, config: config)

let app = Application(
    router: router,
    configuration: .init(address: .hostname(config.host, port: config.port))
)

logger.info("Server ready at http://\(config.host):\(config.port)")
try await app.runService()

// MARK: - Environment Loading

/// Loads environment variables from .env file in the working directory.
func loadDotEnv() {
    let fileManager = FileManager.default
    let dotenvPath = ".env"

    // Try to read .env from current working directory
    guard fileManager.fileExists(atPath: dotenvPath) else {
        return
    }

    guard let content = try? String(contentsOfFile: dotenvPath, encoding: .utf8) else {
        return
    }

    let lines = content.split(separator: "\n", omittingEmptySubsequences: false)
    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        // Skip empty lines and comments
        guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else {
            continue
        }

        // Parse KEY=VALUE format
        let parts = trimmed.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2 else {
            continue
        }

        let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
        var value = String(parts[1]).trimmingCharacters(in: .whitespaces)

        // Remove surrounding quotes if present
        if (value.hasPrefix("\"") && value.hasSuffix("\""))
            || (value.hasPrefix("'") && value.hasSuffix("'"))
        {
            value = String(value.dropFirst().dropLast())
        }

        setenv(key, value, 1)
    }
}

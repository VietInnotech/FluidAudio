import Foundation
import Hummingbird
import Logging
import NIOCore

let config = ServerConfig.fromEnvironment()

var logger = Logger(label: "FluidAudioServer")
logger.logLevel = .info

logger.info("Starting FluidAudio ASR Server")
logger.info("Host: \(config.host)")
logger.info("Port: \(config.port)")
logger.info("Max upload: \(config.maxUploadBytes / 1024 / 1024) MB")
logger.info("Max audio duration: \(config.maxAudioSeconds)s")
logger.info("API key: \(config.apiKey != nil ? "configured" : "disabled (open access)")")

let service = TranscriptionService(config: config)

let router = Router()

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

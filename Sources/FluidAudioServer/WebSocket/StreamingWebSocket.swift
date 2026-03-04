import FluidAudio
import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdWebSocket
import Logging
import NIOCore

// MARK: - WebSocket Router

/// Create the WebSocket router for streaming audio transcription.
///
/// Endpoint: `ws://host:port/v1/audio/stream`
///
/// **Protocol:**
///
/// Client → Server (text):
/// ```json
/// {"type":"session.start","model":"fluidaudio-parakeet-v3","language":"en","encoding":"pcm_s16le","sample_rate":16000}
/// ```
/// Client → Server (binary): Raw PCM audio data (16kHz mono)
/// Client → Server (text): `{"type":"session.stop"}`
///
/// Server → Client (text): `{"type":"session.created","session_id":"..."}`
/// Server → Client (text): `{"type":"transcription.partial","text":"..."}`
/// Server → Client (text): `{"type":"transcription.final","text":"...","duration":...,"processing_time":...}`
/// Server → Client (text): `{"type":"session.stopped","total_duration":...}`
/// Server → Client (text): `{"type":"error","message":"..."}`
func createStreamingWebSocketRouter(
    service: TranscriptionService,
    config: ServerConfig
) -> Router<BasicWebSocketRequestContext> {
    let wsRouter = Router(context: BasicWebSocketRequestContext.self)

    wsRouter.ws("/v1/audio/stream") { request, _ in
        // Authenticate during WebSocket upgrade if API key is configured
        if let apiKey = config.apiKey {
            // Check Authorization: Bearer <key> header
            let authHeader = request.headers[.authorization]
            if let authHeader,
                authHeader.hasPrefix("Bearer "),
                String(authHeader.dropFirst("Bearer ".count)) == apiKey
            {
                return .upgrade([:])
            }
            // Check ?token=<key> query parameter
            let query = request.uri.query ?? ""
            if let token = parseQueryParam("token", from: query),
                token == apiKey
            {
                return .upgrade([:])
            }
            return .dontUpgrade
        }
        return .upgrade([:])
    } onUpgrade: { inbound, outbound, context in
        await StreamingSessionHandler.run(
            inbound: inbound,
            outbound: outbound,
            service: service,
            config: config,
            logger: context.logger
        )
    }

    return wsRouter
}

// MARK: - Session Handler

/// Handles a single WebSocket streaming transcription session.
private enum StreamingSessionHandler {

    /// Main session loop: receives control messages and audio data, sends transcription results.
    static func run(
        inbound: WebSocketInboundStream,
        outbound: WebSocketOutboundWriter,
        service: TranscriptionService,
        config: ServerConfig,
        logger: Logger
    ) async {
        let sessionId = UUID().uuidString
        var logger = logger
        logger[metadataKey: "ws_session"] = "\(sessionId)"

        // Session state
        var audioBuffer: [Float] = []
        var modelID: ModelID = .parakeetV3
        var language: String?
        var encoding = "pcm_s16le"
        var interimResults = true
        var isStarted = false
        var lastInterimSampleCount = 0
        let interimIntervalSamples = 48_000  // 3 seconds at 16kHz

        // Send session.created
        await send(.sessionCreated(sessionId: sessionId), to: outbound)
        logger.info("ws connected")

        do {
            for try await message in inbound.messages(maxSize: 10 * 1024 * 1024) {
                switch message {
                case .text(let text):
                    guard let data = text.data(using: .utf8),
                        let msg = try? JSONDecoder().decode(ClientMessage.self, from: data)
                    else {
                        await send(.error(message: "Invalid JSON"), to: outbound)
                        continue
                    }

                    switch msg.type {
                    case "session.start":
                        guard !isStarted else {
                            await send(.error(message: "Session already started"), to: outbound)
                            continue
                        }

                        if let m = msg.model, let id = ModelID(rawValue: m) {
                            modelID = id
                        }
                        language = msg.language
                        encoding = msg.encoding ?? "pcm_s16le"
                        interimResults = msg.interim_results ?? true

                        guard ["pcm_s16le", "pcm_f32le"].contains(encoding) else {
                            await send(
                                .error(
                                    message:
                                        "Unsupported encoding '\(encoding)'. Use pcm_s16le or pcm_f32le"
                                ),
                                to: outbound
                            )
                            return
                        }

                        if let sr = msg.sample_rate, sr != 16_000 {
                            await send(
                                .error(message: "Only 16000 Hz sample rate is supported"),
                                to: outbound
                            )
                            return
                        }

                        do {
                            try await service.acquireForStreaming(model: modelID)
                        } catch {
                            await send(
                                .error(message: error.localizedDescription),
                                to: outbound
                            )
                            return
                        }

                        isStarted = true
                        logger.info(
                            "streaming: model=\(modelID.rawValue) lang=\(language ?? "auto") enc=\(encoding)"
                        )

                    case "session.stop":
                        guard isStarted else {
                            await send(.error(message: "Session not started"), to: outbound)
                            continue
                        }

                        let audioDuration = Double(audioBuffer.count) / 16_000.0
                        logger.info(
                            "stop: \(String(format: "%.1f", audioDuration))s buffered"
                        )

                        if !audioBuffer.isEmpty {
                            do {
                                let result = try await service.transcribeForStream(
                                    audioSamples: audioBuffer,
                                    model: modelID,
                                    language: language
                                )
                                await send(
                                    .final(
                                        text: result.text,
                                        duration: result.duration,
                                        processingTime: result.processingTime
                                    ),
                                    to: outbound
                                )
                                let rtfx = result.duration / max(result.processingTime, 0.001)
                                let doneMsg =
                                    "final: \(String(format: "%.1f", result.duration))s → "
                                    + "\(String(format: "%.3f", result.processingTime))s "
                                    + "(RTFx \(String(format: "%.1f", rtfx))x) "
                                    + "\(result.text.count) chars"
                                logger.info("\(doneMsg)")
                            } catch {
                                await send(
                                    .error(
                                        message:
                                            "Transcription failed: \(error.localizedDescription)"
                                    ),
                                    to: outbound
                                )
                            }
                        }

                        let totalDuration = Double(audioBuffer.count) / 16_000.0
                        await send(.stopped(totalDuration: totalDuration), to: outbound)
                        await service.releaseStreaming()
                        isStarted = false
                        logger.info("session stopped")
                        return

                    default:
                        await send(
                            .error(message: "Unknown message type: '\(msg.type)'"),
                            to: outbound
                        )
                    }

                case .binary(var buffer):
                    guard isStarted else {
                        await send(
                            .error(message: "Send session.start before audio data"),
                            to: outbound
                        )
                        continue
                    }

                    let samples = decodePCMAudio(&buffer, encoding: encoding)
                    audioBuffer.append(contentsOf: samples)

                    // Send interim transcription when enough new audio has accumulated
                    let newSamples = audioBuffer.count - lastInterimSampleCount
                    if interimResults && newSamples >= interimIntervalSamples {
                        do {
                            let result = try await service.transcribeForStream(
                                audioSamples: audioBuffer,
                                model: modelID,
                                language: language
                            )
                            await send(.partial(text: result.text), to: outbound)
                            lastInterimSampleCount = audioBuffer.count
                        } catch {
                            logger.warning("interim transcription failed: \(error)")
                        }
                    }
                }
            }
        } catch {
            logger.error("ws error: \(error)")
            await send(
                .error(message: "Connection error: \(error.localizedDescription)"),
                to: outbound
            )
        }

        // Cleanup on unexpected disconnect
        if isStarted {
            await service.releaseStreaming()
            logger.info("session released (disconnected)")
        }
    }

    // MARK: - JSON Helpers

    private static func send(_ message: ServerMessage, to outbound: WebSocketOutboundWriter) async {
        guard let data = try? JSONEncoder().encode(message),
            let string = String(data: data, encoding: .utf8)
        else { return }
        try? await outbound.write(.text(string))
    }
}

// MARK: - Protocol Types

/// Client → Server message (all fields optional except type).
private struct ClientMessage: Decodable, Sendable {
    let type: String
    let model: String?
    let language: String?
    let encoding: String?
    let sample_rate: Int?
    let interim_results: Bool?
}

/// Server → Client message.
private enum ServerMessage: Encodable, Sendable {
    case sessionCreated(sessionId: String)
    case partial(text: String)
    case final(text: String, duration: Double, processingTime: Double)
    case stopped(totalDuration: Double)
    case error(message: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case session_id
        case text
        case duration
        case processing_time
        case total_duration
        case message
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .sessionCreated(let sessionId):
            try container.encode("session.created", forKey: .type)
            try container.encode(sessionId, forKey: .session_id)
        case .partial(let text):
            try container.encode("transcription.partial", forKey: .type)
            try container.encode(text, forKey: .text)
        case .final(let text, let duration, let processingTime):
            try container.encode("transcription.final", forKey: .type)
            try container.encode(text, forKey: .text)
            try container.encode(duration, forKey: .duration)
            try container.encode(processingTime, forKey: .processing_time)
        case .stopped(let totalDuration):
            try container.encode("session.stopped", forKey: .type)
            try container.encode(totalDuration, forKey: .total_duration)
        case .error(let message):
            try container.encode("error", forKey: .type)
            try container.encode(message, forKey: .message)
        }
    }
}

// MARK: - PCM Decoding

/// Decode raw PCM bytes from a WebSocket binary frame into Float32 samples.
///
/// Supports:
/// - `pcm_s16le`: 16-bit signed integer, little-endian (most common for streaming)
/// - `pcm_f32le`: 32-bit float, little-endian
private func decodePCMAudio(_ buffer: inout ByteBuffer, encoding: String) -> [Float] {
    let byteCount = buffer.readableBytes
    guard byteCount > 0 else { return [] }

    switch encoding {
    case "pcm_f32le":
        let sampleCount = byteCount / 4
        guard sampleCount > 0 else { return [] }
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            guard let bits = buffer.readInteger(endianness: .little, as: UInt32.self) else { break }
            samples[i] = Float(bitPattern: bits)
        }
        return samples

    case "pcm_s16le":
        let sampleCount = byteCount / 2
        guard sampleCount > 0 else { return [] }
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            guard let value = buffer.readInteger(endianness: .little, as: Int16.self) else { break }
            samples[i] = Float(value) / 32768.0
        }
        return samples

    default:
        return []
    }
}

// MARK: - Query String Helper

/// Extract a query parameter value from a raw query string.
private func parseQueryParam(_ name: String, from query: String) -> String? {
    let pairs = query.split(separator: "&")
    for pair in pairs {
        let parts = pair.split(separator: "=", maxSplits: 1)
        guard parts.count == 2 else { continue }
        if parts[0] == name {
            return String(parts[1])
        }
    }
    return nil
}

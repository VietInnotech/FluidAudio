import FluidAudio
import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdWebSocket
import Logging
import NIOCore

// MARK: - Session Config

/// Session configuration parsed from WebSocket URL query parameters.
///
/// All options are specified at connection time — no JSON handshake required.
///
/// Example:
/// ```
/// ws://localhost:8080/v1/audio/stream
///   ?model=fluidaudio-parakeet-v3
///   &language=en
///   &encoding=pcm_s16le
///   &interim_results=true
///   &endpointing=700
/// ```
struct StreamingSessionConfig: Sendable {
    let model: ModelID
    let language: String?
    /// `pcm_s16le` (default) or `pcm_f32le`. Must be 16 kHz mono.
    let encoding: String
    /// Whether to emit `Results` with `is_final:false` during active speech. Default: `true`.
    let interimResults: Bool
    /// Milliseconds of silence after speech that triggers utterance end. Default: 700 ms.
    let endpointingMs: Int

    static func from(query: String) -> StreamingSessionConfig {
        let modelRaw = parseQueryParam("model", from: query) ?? ModelID.parakeetV3.rawValue
        let model = ModelID(rawValue: modelRaw) ?? .parakeetV3
        let language = parseQueryParam("language", from: query)
        let encoding = parseQueryParam("encoding", from: query) ?? "pcm_s16le"
        let interimResults = parseQueryParam("interim_results", from: query) != "false"
        let endpointingMs = Int(parseQueryParam("endpointing", from: query) ?? "") ?? 700
        return StreamingSessionConfig(
            model: model,
            language: language,
            encoding: encoding,
            interimResults: interimResults,
            endpointingMs: endpointingMs
        )
    }
}

// MARK: - WebSocket Router

/// Create the Deepgram-compatible WebSocket router for streaming ASR.
///
/// **Endpoint:** `ws://host:port/v1/audio/stream`
///
/// **Authentication:**
/// - Header `Authorization: Bearer <key>` on the HTTP upgrade request
/// - Query param `?token=<key>` (useful for clients that can't set headers)
///
/// **Query parameters (all optional):**
/// | Param | Default | Description |
/// |-------|---------|-------------|
/// | `model` | `fluidaudio-parakeet-v3` | ASR model ID |
/// | `language` | _(auto)_ | ISO 639-1 code, e.g. `en` |
/// | `encoding` | `pcm_s16le` | `pcm_s16le` or `pcm_f32le` |
/// | `interim_results` | `true` | Emit partial results while speaking |
/// | `endpointing` | `700` | Silence (ms) to finalize an utterance |
/// | `token` | — | API key alternative to Authorization header |
///
/// **Client → Server:**
/// - **Binary frames** — raw 16 kHz mono PCM. Send immediately after connecting.
/// - **`{"type":"Finalize"}`** — force-flush the current utterance buffer.
/// - **`{"type":"KeepAlive"}`** — prevent idle timeout.
/// - Closing the WebSocket ends the session naturally.
///
/// **Server → Client:**
/// ```json
/// {"type":"Metadata","request_id":"...","model":"fluidaudio-parakeet-v3"}
/// {"type":"SpeechStarted","channel_index":[0],"timestamp":1.2}
/// {"type":"Results","channel_index":[0,1],"start":0.0,"duration":1.8,
///  "is_final":false,"speech_final":false,
///  "channel":{"alternatives":[{"transcript":"hello wor","confidence":1.0,"words":[]}]}}
/// {"type":"Results","channel_index":[0,1],"start":0.0,"duration":3.2,
///  "is_final":true,"speech_final":true,
///  "channel":{"alternatives":[{"transcript":"hello world","confidence":1.0,"words":[]}]}}
/// {"type":"UtteranceEnd","channel":[0],"last_word_end":3.2}
/// {"type":"Error","description":"...","variant":"..."}
/// ```
func createStreamingWebSocketRouter(
    service: TranscriptionService,
    config: ServerConfig
) -> Router<BasicWebSocketRequestContext> {
    let wsRouter = Router(context: BasicWebSocketRequestContext.self)

    wsRouter.ws("/v1/audio/stream") { request, _ in
        if let apiKey = config.apiKey {
            let authHeader = request.headers[.authorization]
            if let authHeader,
                authHeader.hasPrefix("Bearer "),
                String(authHeader.dropFirst("Bearer ".count)) == apiKey
            {
                return .upgrade([:])
            }
            let query = request.uri.query ?? ""
            if let token = parseQueryParam("token", from: query), token == apiKey {
                return .upgrade([:])
            }
            return .dontUpgrade
        }
        return .upgrade([:])
    } onUpgrade: { inbound, outbound, context in
        let query = context.request.uri.query ?? ""
        let sessionConfig = StreamingSessionConfig.from(query: query)
        await StreamingSessionHandler.run(
            inbound: inbound,
            outbound: outbound,
            sessionConfig: sessionConfig,
            service: service,
            logger: context.logger
        )
    }

    return wsRouter
}

// MARK: - Session Handler

/// Handles a single WebSocket streaming transcription session.
///
/// Uses the trained Silero VAD model (via `VadManager`) for speech activity detection instead of manual
/// energy-based thresholding. This provides more robust endpointing behavior, especially in noisy environments.
/// Audio is processed in-stream, with speech buffered and finalized once endpointing silence duration is reached.
private enum StreamingSessionHandler {

    private static let sampleRate = VadManager.sampleRate
    /// Minimum speech samples for transcription (250 ms).
    private static let minUtteranceSamples = sampleRate / 4
    /// Emit an interim result every 2 s of accumulated speech.
    private static let interimIntervalSamples = sampleRate * 2

    static func run(
        inbound: WebSocketInboundStream,
        outbound: WebSocketOutboundWriter,
        sessionConfig: StreamingSessionConfig,
        service: TranscriptionService,
        logger: Logger
    ) async {
        let requestId = UUID().uuidString
        var logger = logger
        logger[metadataKey: "ws"] = "\(requestId.prefix(8))"

        // Send Metadata immediately — client can start sending audio right after.
        await sendMsg(.metadata(requestId: requestId, model: sessionConfig.model.rawValue), to: outbound)
        logger.info("ws connected: model=\(sessionConfig.model.rawValue) lang=\(sessionConfig.language ?? "auto") enc=\(sessionConfig.encoding) endpointing=\(sessionConfig.endpointingMs)ms")

        do {
            try await service.acquireForStreaming(model: sessionConfig.model)
        } catch {
            await sendMsg(.error(description: error.localizedDescription, variant: "service_unavailable"), to: outbound)
            logger.warning("ws rejected: \(error)")
            return
        }

        // Initialize VAD for speech detection instead of manual energy thresholding
        let vad: VadManager
        do {
            vad = try await VadManager(config: VadConfig(defaultThreshold: 0.5))
        } catch {
            await sendMsg(.error(description: "VAD initialization failed: \(error.localizedDescription)", variant: "initialization_error"), to: outbound)
            await service.releaseStreaming()
            logger.warning("VAD init failed: \(error)")
            return
        }

        let silentSamplesToEndpoint = sampleRate * sessionConfig.endpointingMs / 1000

        // Per-utterance state
        var speechBuffer: [Float] = []
        var bufferStartSample = 0
        var speechStartSample = 0
        var totalSamples = 0
        var hasSpeech = false
        var speechSamples = 0
        var silentAfterSpeech = 0
        var lastInterimSpeechSamples = 0
        var utteranceCount = 0
        var vadStreamState = VadStreamState.initial()

        do {
            for try await message in inbound.messages(maxSize: 4 * 1024 * 1024) {
                switch message {
                case .binary(var buffer):
                    let samples = decodePCMAudio(&buffer, encoding: sessionConfig.encoding)
                    guard !samples.isEmpty else { continue }
                    totalSamples += samples.count
                    speechBuffer.append(contentsOf: samples)

                    // Use VadManager to detect speech activity
                    let vadResult = try await vad.processStreamingChunk(
                        samples,
                        state: vadStreamState,
                        config: .default,
                        returnSeconds: false,
                        timeResolution: 1
                    )
                    vadStreamState = vadResult.state
                    let isSpeechActive = vadResult.probability >= 0.5

                    if isSpeechActive {
                        if !hasSpeech {
                            hasSpeech = true
                            speechStartSample = totalSamples - samples.count
                            speechSamples = 0
                            silentAfterSpeech = 0
                            lastInterimSpeechSamples = 0
                            let ts = Double(speechStartSample) / Double(sampleRate)
                            await sendMsg(.speechStarted(timestamp: ts), to: outbound)
                            logger.debug("speech started at \(String(format: "%.2f", ts))s")
                        }
                        speechSamples += samples.count
                        silentAfterSpeech = 0
                    } else if hasSpeech {
                        silentAfterSpeech += samples.count
                    }

                    // Endpointing: enough silence → finalize utterance
                    if hasSpeech && silentAfterSpeech >= silentSamplesToEndpoint {
                        utteranceCount += 1
                        await finalizeUtterance(
                            &speechBuffer,
                            bufferStartSample: bufferStartSample,
                            speechStartSample: speechStartSample,
                            trailingsilence: silentAfterSpeech,
                            totalSamples: totalSamples,
                            index: utteranceCount,
                            config: sessionConfig,
                            service: service,
                            outbound: outbound,
                            logger: logger
                        )
                        bufferStartSample = totalSamples
                        speechBuffer.removeAll(keepingCapacity: true)
                        hasSpeech = false
                        speechSamples = 0
                        silentAfterSpeech = 0
                        lastInterimSpeechSamples = 0
                        vadStreamState = VadStreamState.initial()
                        continue
                    }

                    // Interim results every interimIntervalSamples of speech
                    guard sessionConfig.interimResults && hasSpeech else { continue }
                    let newSpeech = speechSamples - lastInterimSpeechSamples
                    guard newSpeech >= interimIntervalSamples else { continue }
                    let start = Double(speechStartSample) / Double(sampleRate)
                    let duration = Double(totalSamples - speechStartSample) / Double(sampleRate)
                    let relStart = speechStartSample - bufferStartSample
                    guard relStart >= 0 && relStart < speechBuffer.count else { continue }
                    let slice = Array(speechBuffer[relStart...])
                    await emitInterim(
                        slice,
                        start: start,
                        duration: duration,
                        config: sessionConfig,
                        service: service,
                        outbound: outbound,
                        logger: logger
                    )
                    lastInterimSpeechSamples = speechSamples

                case .text(let text):
                    guard let data = text.data(using: .utf8),
                        let ctrl = try? JSONDecoder().decode(ControlMessage.self, from: data)
                    else {
                        logger.debug("ws: unrecognized text frame")
                        continue
                    }
                    switch ctrl.type {
                    case "Finalize":
                        guard hasSpeech else { continue }
                        utteranceCount += 1
                        logger.info("Finalize: flushing utterance \(utteranceCount)")
                        await finalizeUtterance(
                            &speechBuffer,
                            bufferStartSample: bufferStartSample,
                            speechStartSample: speechStartSample,
                            trailingsilence: silentAfterSpeech,
                            totalSamples: totalSamples,
                            index: utteranceCount,
                            config: sessionConfig,
                            service: service,
                            outbound: outbound,
                            logger: logger
                        )
                        bufferStartSample = totalSamples
                        speechBuffer.removeAll(keepingCapacity: true)
                        hasSpeech = false
                        speechSamples = 0
                        silentAfterSpeech = 0
                        lastInterimSpeechSamples = 0
                    case "KeepAlive":
                        logger.debug("ws keepalive")
                    default:
                        logger.debug("ws: unknown control '\(ctrl.type)'")
                    }
                }
            }
        } catch {
            logger.error("ws error: \(error)")
            await sendMsg(.error(description: error.localizedDescription, variant: "connection_error"), to: outbound)
        }

        // Flush any remaining speech on disconnect
        if hasSpeech && speechBuffer.count >= minUtteranceSamples {
            utteranceCount += 1
            await finalizeUtterance(
                &speechBuffer,
                bufferStartSample: bufferStartSample,
                speechStartSample: speechStartSample,
                trailingsilence: silentAfterSpeech,
                totalSamples: totalSamples,
                index: utteranceCount,
                config: sessionConfig,
                service: service,
                outbound: outbound,
                logger: logger
            )
        }

        await service.releaseStreaming()
        let totalDuration = Double(totalSamples) / Double(sampleRate)
        logger.info(
            "ws session ended: \(String(format: "%.1f", totalDuration))s audio \(utteranceCount) utterances"
        )
    }

    // MARK: - Utterance finalization

    private static func finalizeUtterance(
        _ speechBuffer: inout [Float],
        bufferStartSample: Int,
        speechStartSample: Int,
        trailingsilence: Int,
        totalSamples: Int,
        index: Int,
        config: StreamingSessionConfig,
        service: TranscriptionService,
        outbound: WebSocketOutboundWriter,
        logger: Logger
    ) async {
        let relStart = max(0, speechStartSample - bufferStartSample)
        guard relStart < speechBuffer.count else { return }
        // Trim trailing silence, keeping a short pad (100 ms) so the last word decodes cleanly
        let pad = sampleRate / 10
        let trimEnd = speechBuffer.count - max(0, trailingsilence - pad)
        let sliceEnd = max(relStart + minUtteranceSamples, min(speechBuffer.count, trimEnd))
        guard sliceEnd > relStart else { return }
        let samples = Array(speechBuffer[relStart..<sliceEnd])
        guard samples.count >= minUtteranceSamples else { return }

        let start = Double(speechStartSample) / Double(sampleRate)
        let duration = Double(samples.count) / Double(sampleRate)

        logger.debug(
            "utterance \(index): \(String(format: "%.2f", start))–\(String(format: "%.2f", start + duration))s"
        )

        do {
            let result = try await service.transcribeForStream(
                audioSamples: samples,
                model: config.model,
                language: config.language
            )
            let transcript = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !transcript.isEmpty else { return }

            let rtfx = duration / max(result.processingTime, 0.001)
            let rtfxStr = String(format: "%.1f", rtfx)
            let durStr = String(format: "%.1f", duration)
            logger.info("utterance \(index): \"\(transcript.prefix(80))\" [\(durStr)s RTFx \(rtfxStr)x]")

            await sendMsg(
                .results(transcript: transcript, start: start, duration: duration, isFinal: true, speechFinal: true),
                to: outbound
            )
            await sendMsg(.utteranceEnd(lastWordEnd: start + duration), to: outbound)
        } catch {
            logger.warning("utterance \(index) failed: \(error)")
            await sendMsg(.error(description: error.localizedDescription, variant: "transcription_error"), to: outbound)
        }
    }

    // MARK: - Interim emission

    private static func emitInterim(
        _ samples: [Float],
        start: Double,
        duration: Double,
        config: StreamingSessionConfig,
        service: TranscriptionService,
        outbound: WebSocketOutboundWriter,
        logger: Logger
    ) async {
        guard samples.count >= minUtteranceSamples else { return }
        do {
            let result = try await service.transcribeForStream(
                audioSamples: samples,
                model: config.model,
                language: config.language
            )
            let transcript = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !transcript.isEmpty else { return }
            logger.debug("interim: \"\(transcript.prefix(60))\"")
            await sendMsg(
                .results(transcript: transcript, start: start, duration: duration, isFinal: false, speechFinal: false),
                to: outbound
            )
        } catch {
            logger.debug("interim skipped: \(error)")
        }
    }

    // MARK: - Send helper

    private static func sendMsg(_ message: ServerMessage, to outbound: WebSocketOutboundWriter) async {
        guard let data = try? JSONEncoder().encode(message),
            let text = String(data: data, encoding: .utf8)
        else { return }
        try? await outbound.write(.text(text))
    }
}

// MARK: - Audio Helpers

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
    default:  // pcm_s16le
        let sampleCount = byteCount / 2
        guard sampleCount > 0 else { return [] }
        var samples = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount {
            guard let value = buffer.readInteger(endianness: .little, as: Int16.self) else { break }
            samples[i] = Float(value) / 32768.0
        }
        return samples
    }
}

// MARK: - Query String

private func parseQueryParam(_ name: String, from query: String) -> String? {
    for pair in query.split(separator: "&") {
        let parts = pair.split(separator: "=", maxSplits: 1)
        guard parts.count == 2, parts[0] == name else { continue }
        return String(parts[1])
    }
    return nil
}

// MARK: - Protocol Types

private struct ControlMessage: Decodable, Sendable {
    let type: String
}

private enum ServerMessage: Encodable, Sendable {
    case metadata(requestId: String, model: String)
    case speechStarted(timestamp: Double)
    case results(transcript: String, start: Double, duration: Double, isFinal: Bool, speechFinal: Bool)
    case utteranceEnd(lastWordEnd: Double)
    case error(description: String, variant: String)
}

extension ServerMessage {

    private struct Alternative: Encodable {
        let transcript: String
        let confidence: Float = 1.0
        let words: [String] = []
    }

    private struct Channel: Encodable {
        let alternatives: [Alternative]
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: Keys.self)
        switch self {
        case .metadata(let requestId, let model):
            try c.encode("Metadata", forKey: .type)
            try c.encode(requestId, forKey: .request_id)
            try c.encode(model, forKey: .model)
        case .speechStarted(let timestamp):
            try c.encode("SpeechStarted", forKey: .type)
            try c.encode([0], forKey: .channel_index)
            try c.encode(timestamp, forKey: .timestamp)
        case .results(let transcript, let start, let duration, let isFinal, let speechFinal):
            try c.encode("Results", forKey: .type)
            try c.encode([0, 1], forKey: .channel_index)
            try c.encode(start, forKey: .start)
            try c.encode(duration, forKey: .duration)
            try c.encode(isFinal, forKey: .is_final)
            try c.encode(speechFinal, forKey: .speech_final)
            try c.encode(Channel(alternatives: [Alternative(transcript: transcript)]), forKey: .channel)
        case .utteranceEnd(let lastWordEnd):
            try c.encode("UtteranceEnd", forKey: .type)
            try c.encode([0], forKey: .channel)
            try c.encode(lastWordEnd, forKey: .last_word_end)
        case .error(let description, let variant):
            try c.encode("Error", forKey: .type)
            try c.encode(description, forKey: .description)
            try c.encode(variant, forKey: .variant)
        }
    }

    private enum Keys: String, CodingKey {
        case type, request_id, model, channel_index, timestamp
        case start, duration, is_final, speech_final, channel
        case last_word_end, description, variant
    }
}

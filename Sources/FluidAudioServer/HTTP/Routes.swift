import FluidAudio
import Foundation
import HTTPTypes
import Hummingbird
import MultipartKit
import NIOCore

/// Register all API routes on the given router.
func registerRoutes(
    on router: Router<BasicRequestContext>,
    service: TranscriptionService,
    config: ServerConfig
) {
    // List models
    router.get("v1/models") { _, _ -> Response in
        let list = ModelsListResponse.build()
        let data = try JSONEncoder().encode(list)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    // Transcription endpoint
    router.post("v1/audio/transcriptions") { request, context -> Response in
        // Validate content type is multipart/form-data
        guard let contentType = request.headers[.contentType],
            contentType.contains("multipart/form-data")
        else {
            throw ServerError.unsupportedMediaType(
                "Expected multipart/form-data content type"
            )
        }

        // Extract boundary from content type
        guard let boundary = extractBoundary(from: contentType) else {
            throw ServerError.badRequest("Missing boundary in multipart content type")
        }

        // Collect the full body with upload size limit
        let body = try await request.body.collect(upTo: config.maxUploadBytes)

        // Parse multipart form data
        let parts = try parseMultipart(body: body, boundary: boundary)

        // Extract required fields
        guard let fileData = parts.file else {
            throw ServerError.badRequest("Missing required field: file")
        }
        guard let modelString = parts.model else {
            throw ServerError.badRequest("Missing required field: model")
        }
        guard let modelID = ModelID(rawValue: modelString) else {
            throw ServerError.modelNotFound(modelString)
        }

        let responseFormat = parts.responseFormat ?? "json"
        guard ["json", "text", "verbose_json"].contains(responseFormat) else {
            throw ServerError.badRequest(
                "Invalid response_format '\(responseFormat)'. Supported: json, text, verbose_json"
            )
        }

        // Write audio to temp file for AudioConverter
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(parts.fileExtension ?? "wav")

        defer { try? FileManager.default.removeItem(at: tempURL) }

        try fileData.write(to: tempURL)

        // Convert audio to 16kHz mono Float32
        let audioConverter = AudioConverter()
        let audioSamples: [Float]
        do {
            audioSamples = try audioConverter.resampleAudioFile(tempURL)
        } catch {
            throw ServerError.badRequest("Failed to decode audio file: \(error.localizedDescription)")
        }

        guard !audioSamples.isEmpty else {
            throw ServerError.badRequest("Audio file is empty or could not be decoded")
        }

        // Transcribe
        let result = try await service.transcribe(
            audioSamples: audioSamples,
            model: modelID,
            language: parts.language
        )

        // Format response
        return try formatResponse(result: result, format: responseFormat)
    }
}

// MARK: - Multipart Parsing

/// Parsed multipart fields relevant to the transcription endpoint.
private struct ParsedMultipartFields {
    var file: Data?
    var fileExtension: String?
    var model: String?
    var language: String?
    var responseFormat: String?
}

/// Extract boundary string from Content-Type header.
private func extractBoundary(from contentType: String) -> String? {
    // Content-Type: multipart/form-data; boundary=----WebKitFormBoundary...
    let parts = contentType.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
    for part in parts {
        if part.lowercased().hasPrefix("boundary=") {
            var boundary = String(part.dropFirst("boundary=".count))
            // Remove surrounding quotes if present
            if boundary.hasPrefix("\"") && boundary.hasSuffix("\"") {
                boundary = String(boundary.dropFirst().dropLast())
            }
            return boundary
        }
    }
    return nil
}

/// Parse multipart form data by splitting on the boundary and extracting headers/body per part.
private func parseMultipart(body: ByteBuffer, boundary: String) throws -> ParsedMultipartFields {
    var fields = ParsedMultipartFields()

    // Collect all parts by parsing the raw bytes
    var bodyData = body
    guard let bytes = bodyData.readBytes(length: bodyData.readableBytes) else {
        throw ServerError.badRequest("Empty request body")
    }

    // Simple multipart parsing: split by boundary
    let boundaryBytes = "--\(boundary)"
    let content = String(decoding: bytes, as: UTF8.self)
    let rawParts = content.components(separatedBy: boundaryBytes)

    for rawPart in rawParts {
        let trimmed = rawPart.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != "--" else { continue }

        // Split headers from body by double newline
        let headerBodySplit: (headers: String, body: String)
        if let range = trimmed.range(of: "\r\n\r\n") {
            headerBodySplit = (
                String(trimmed[trimmed.startIndex..<range.lowerBound]),
                String(trimmed[range.upperBound...])
            )
        } else if let range = trimmed.range(of: "\n\n") {
            headerBodySplit = (
                String(trimmed[trimmed.startIndex..<range.lowerBound]),
                String(trimmed[range.upperBound...])
            )
        } else {
            continue
        }

        let headers = headerBodySplit.headers
        let bodyContent = headerBodySplit.body

        // Extract field name from Content-Disposition
        guard let nameMatch = extractFieldName(from: headers) else { continue }

        switch nameMatch {
        case "file":
            // For file data, we must reconstruct the raw binary instead of using string conversion
            let fileExtension = extractFilename(from: headers)
                .flatMap { URL(fileURLWithPath: $0).pathExtension }
            fields.fileExtension = fileExtension

            // Re-extract raw bytes for the file part
            fields.file = extractFileBytes(
                from: bytes, boundary: boundaryBytes, fieldName: "file"
            )

        case "model":
            fields.model = bodyContent.trimmingCharacters(in: .whitespacesAndNewlines)

        case "language":
            fields.language = bodyContent.trimmingCharacters(in: .whitespacesAndNewlines)

        case "response_format":
            fields.responseFormat = bodyContent.trimmingCharacters(in: .whitespacesAndNewlines)

        default:
            break
        }
    }

    return fields
}

/// Extract the field name from Content-Disposition header.
private func extractFieldName(from headers: String) -> String? {
    // Content-Disposition: form-data; name="file"; filename="audio.wav"
    guard headers.contains("name=") else { return nil }

    let pattern = #"name="([^"]+)""#
    guard let regex = try? NSRegularExpression(pattern: pattern),
        let match = regex.firstMatch(
            in: headers,
            range: NSRange(headers.startIndex..., in: headers)
        ),
        let nameRange = Range(match.range(at: 1), in: headers)
    else {
        return nil
    }

    return String(headers[nameRange])
}

/// Extract filename from Content-Disposition header.
private func extractFilename(from headers: String) -> String? {
    let pattern = #"filename="([^"]+)""#
    guard let regex = try? NSRegularExpression(pattern: pattern),
        let match = regex.firstMatch(
            in: headers,
            range: NSRange(headers.startIndex..., in: headers)
        ),
        let range = Range(match.range(at: 1), in: headers)
    else {
        return nil
    }
    return String(headers[range])
}

/// Extract raw file bytes from the multipart body for a specific field.
///
/// This avoids string encoding issues by working directly with the byte array.
private func extractFileBytes(from bytes: [UInt8], boundary: String, fieldName: String) -> Data? {
    let content = bytes
    let boundaryData = Array(boundary.utf8)

    // Find the boundary that precedes the field
    var searchStart = 0
    while searchStart < content.count {
        guard let boundaryPos = findSequence(boundaryData, in: content, from: searchStart) else {
            break
        }

        // Find the double CRLF after this boundary (end of headers)
        let afterBoundary = boundaryPos + boundaryData.count
        guard let headerEnd = findDoubleCRLF(in: content, from: afterBoundary) else {
            searchStart = afterBoundary
            continue
        }

        // Check if this part's headers contain our field name
        let headerBytes = Array(content[afterBoundary..<headerEnd])
        let headerStr = String(decoding: headerBytes, as: UTF8.self)

        if let name = extractFieldName(from: headerStr), name == fieldName {
            // Body starts after the double CRLF
            let bodyStart = headerEnd + 4  // skip \r\n\r\n

            // Body ends at the next boundary (minus the \r\n before the boundary)
            guard let nextBoundary = findSequence(boundaryData, in: content, from: bodyStart) else {
                return nil
            }

            // The body ends 2 bytes before the next boundary (\r\n)
            var bodyEnd = nextBoundary
            if bodyEnd >= 2 && content[bodyEnd - 1] == 0x0A && content[bodyEnd - 2] == 0x0D {
                bodyEnd -= 2
            }

            guard bodyStart <= bodyEnd else { return nil }
            return Data(content[bodyStart..<bodyEnd])
        }

        searchStart = afterBoundary
    }

    return nil
}

/// Find a byte sequence in a larger byte array starting from an offset.
private func findSequence(_ needle: [UInt8], in haystack: [UInt8], from start: Int) -> Int? {
    guard needle.count <= haystack.count - start else { return nil }
    let end = haystack.count - needle.count
    for i in start...end {
        if haystack[i..<(i + needle.count)].elementsEqual(needle) {
            return i
        }
    }
    return nil
}

/// Find double CRLF (\r\n\r\n) in byte array.
private func findDoubleCRLF(in bytes: [UInt8], from start: Int) -> Int? {
    let pattern: [UInt8] = [0x0D, 0x0A, 0x0D, 0x0A]
    return findSequence(pattern, in: bytes, from: start)
}

// MARK: - Response Formatting

/// OpenAI-compatible JSON transcription response.
private struct TranscriptionResponse: Codable {
    let text: String
}

/// OpenAI-compatible verbose JSON transcription response.
private struct VerboseTranscriptionResponse: Codable {
    let text: String
    let model: String
    let duration: Double
    let processing_time: Double
}

private func formatResponse(result: TranscriptionResult, format: String) throws -> Response {
    switch format {
    case "text":
        return Response(
            status: .ok,
            headers: [.contentType: "text/plain"],
            body: .init(byteBuffer: ByteBuffer(string: result.text))
        )

    case "verbose_json":
        let verbose = VerboseTranscriptionResponse(
            text: result.text,
            model: result.model,
            duration: result.duration,
            processing_time: result.processingTime
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(verbose)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )

    default:
        let json = TranscriptionResponse(text: result.text)
        let data = try JSONEncoder().encode(json)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }
}

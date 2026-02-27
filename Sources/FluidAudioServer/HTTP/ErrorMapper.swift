import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

/// Structured JSON error response for the API.
struct APIErrorResponse: Codable, Sendable {
    struct ErrorDetail: Codable, Sendable {
        let message: String
        let type: String
        let code: String?
    }

    let error: ErrorDetail
}

/// Maps domain errors to appropriate HTTP responses.
struct ServerError: HTTPResponseError, Sendable {
    let status: HTTPResponse.Status
    let errorType: String
    let message: String
    let code: String?

    init(status: HTTPResponse.Status, type: String, message: String, code: String? = nil) {
        self.status = status
        self.errorType = type
        self.message = message
        self.code = code
    }

    func response(from request: Request, context: some RequestContext) throws -> Response {
        let errorBody = APIErrorResponse(
            error: .init(message: message, type: errorType, code: code)
        )
        let data =
            (try? JSONEncoder().encode(errorBody))
            ?? Data(#"{"error":{"message":"Internal error","type":"server_error"}}"#.utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
    }

    // MARK: - Factory Methods

    static func badRequest(_ message: String) -> ServerError {
        ServerError(status: .badRequest, type: "invalid_request_error", message: message)
    }

    static func unauthorized(_ message: String) -> ServerError {
        ServerError(status: .unauthorized, type: "authentication_error", message: message)
    }

    static func contentTooLarge(_ message: String) -> ServerError {
        ServerError(status: .contentTooLarge, type: "invalid_request_error", message: message)
    }

    static func unsupportedMediaType(_ message: String) -> ServerError {
        ServerError(status: .unsupportedMediaType, type: "invalid_request_error", message: message)
    }

    static func tooManyRequests(_ message: String) -> ServerError {
        ServerError(status: .tooManyRequests, type: "rate_limit_error", message: message)
    }

    static func internalError(_ message: String) -> ServerError {
        ServerError(status: .internalServerError, type: "server_error", message: message)
    }

    static func modelNotFound(_ model: String) -> ServerError {
        ServerError(
            status: .badRequest,
            type: "invalid_request_error",
            message: "Model '\(model)' not found. Use GET /v1/models to list available models.",
            code: "model_not_found"
        )
    }
}

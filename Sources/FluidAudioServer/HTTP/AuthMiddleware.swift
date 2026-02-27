import HTTPTypes
import Hummingbird

/// Middleware that validates Bearer token authentication when an API key is configured.
struct AuthMiddleware: RouterMiddleware {
    typealias Context = BasicRequestContext

    let apiKey: String

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let authorization = request.headers[.authorization]
        guard let authorization else {
            throw HTTPError(.unauthorized, message: "Missing Authorization header")
        }

        let prefix = "Bearer "
        guard authorization.hasPrefix(prefix) else {
            throw HTTPError(.unauthorized, message: "Invalid Authorization header format, expected: Bearer <key>")
        }

        let token = String(authorization.dropFirst(prefix.count))
        guard token == apiKey else {
            throw HTTPError(.unauthorized, message: "Invalid API key")
        }

        return try await next(request, context)
    }
}

import Foundation
import Hummingbird
import Logging

/// Middleware that logs every HTTP request and its response with timing.
///
/// Output format:
/// ```
/// → POST /v1/audio/transcriptions [2.1 MB]
/// ← 200 OK 1.234s
/// ```
struct RequestLoggingMiddleware: RouterMiddleware {
    typealias Context = BasicRequestContext

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let startTime = CFAbsoluteTimeGetCurrent()

        // Format content-length hint if header is present
        let sizeHint: String
        if let contentLength = request.headers[.contentLength], let bytes = Int(contentLength) {
            sizeHint = " [\(formatBytes(bytes))]"
        } else {
            sizeHint = ""
        }

        context.logger.info("→ \(request.method) \(request.uri.path)\(sizeHint)")

        do {
            let response = try await next(request, context)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let level: Logger.Level = response.status.code >= 400 ? .warning : .info
            context.logger.log(
                level: level,
                "← \(response.status.code) \(response.status.reasonPhrase) \(formatDuration(elapsed))"
            )
            return response
        } catch {
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            context.logger.warning("← ERR \(formatDuration(elapsed)) — \(error.localizedDescription)")
            throw error
        }
    }
}

// MARK: - Formatting Helpers

private func formatDuration(_ seconds: Double) -> String {
    seconds < 1 ? "\(Int(seconds * 1000))ms" : String(format: "%.3fs", seconds)
}

private func formatBytes(_ bytes: Int) -> String {
    switch bytes {
    case ..<1024: return "\(bytes) B"
    case ..<(1024 * 1024): return String(format: "%.1f KB", Double(bytes) / 1024)
    default: return String(format: "%.1f MB", Double(bytes) / 1024 / 1024)
    }
}

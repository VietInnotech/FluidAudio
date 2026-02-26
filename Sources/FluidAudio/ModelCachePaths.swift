import Foundation

enum ModelCachePaths {
    static let modelsDirEnvironmentVariable = "FLUIDAUDIO_MODELS_DIR"

    /// Root directory where ASR/VAD/diarization models are stored.
    ///
    /// Priority:
    /// 1. `FLUIDAUDIO_MODELS_DIR` environment variable
    /// 2. `~/Library/Application Support/FluidAudio/Models`
    /// 3. Temporary directory fallback
    static func modelsRootDirectory() -> URL {
        if
            let overrideValue = ProcessInfo.processInfo.environment[modelsDirEnvironmentVariable]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !overrideValue.isEmpty
        {
            let expandedPath = (overrideValue as NSString).expandingTildeInPath
            if expandedPath.hasPrefix("/") {
                return URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
            }

            let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            return URL(fileURLWithPath: expandedPath, relativeTo: cwd).standardizedFileURL
        }

        if
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        {
            return
                appSupport
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }

        return
            FileManager.default.temporaryDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}

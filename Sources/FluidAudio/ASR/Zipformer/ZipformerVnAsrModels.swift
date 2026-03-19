import Foundation
import OnnxRuntimeBindings

public struct ZipformerVnAsrModels {
    public let modelDirectory: URL
    public let encoderSession: ORTSession
    public let decoderSession: ORTSession
    public let joinerSession: ORTSession
    public let tokens: [Int: String]

    private static let logger = AppLogger(category: "ZipformerVnAsrModels")

    public static func load(from directory: URL) throws -> ZipformerVnAsrModels {
        let encoderPath = directory.appendingPathComponent(ModelNames.ZipformerVN.encoderFile)
        let decoderPath = directory.appendingPathComponent(ModelNames.ZipformerVN.decoderFile)
        let joinerPath = directory.appendingPathComponent(ModelNames.ZipformerVN.joinerFile)
        let tokensPath = directory.appendingPathComponent(ModelNames.ZipformerVN.tokensFile)

        for (name, path) in [
            (ModelNames.ZipformerVN.encoderFile, encoderPath),
            (ModelNames.ZipformerVN.decoderFile, decoderPath),
            (ModelNames.ZipformerVN.joinerFile, joinerPath),
            (ModelNames.ZipformerVN.tokensFile, tokensPath),
        ] {
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw AsrModelsError.modelNotFound(name, path)
            }
        }

        let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let sessionOptions = try ORTSessionOptions()
        try sessionOptions.setIntraOpNumThreads(1)
        try sessionOptions.setGraphOptimizationLevel(.all)

        if ORTIsCoreMLExecutionProviderAvailable() {
            _ = try? sessionOptions.appendCoreMLExecutionProvider(
                withOptionsV2: [
                    "EnableOnSubgraphs": "1",
                    "RequireStaticInputShapes": "1",
                ]
            )
        }

        let encoderSession = try ORTSession(env: env, modelPath: encoderPath.path, sessionOptions: sessionOptions)
        let decoderSession = try ORTSession(env: env, modelPath: decoderPath.path, sessionOptions: sessionOptions)
        let joinerSession = try ORTSession(env: env, modelPath: joinerPath.path, sessionOptions: sessionOptions)

        let tokenMap = try loadTokens(from: tokensPath)

        return ZipformerVnAsrModels(
            modelDirectory: directory,
            encoderSession: encoderSession,
            decoderSession: decoderSession,
            joinerSession: joinerSession,
            tokens: tokenMap
        )
    }

    @discardableResult
    public static func download(to directory: URL? = nil, force: Bool = false) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()

        if !force && modelsExist(at: targetDir) {
            return targetDir
        }

        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let releaseURL = URL(
            string:
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                    + "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
        )!

        logger.info("Downloading Zipformer VN release bundle from GitHub...")
        let (archiveURL, response) = try await DownloadUtils.sharedSession.download(from: releaseURL)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw AsrModelsError.downloadFailed("HTTP failure while downloading Zipformer VN release bundle")
        }

        let fileManager = FileManager.default
        let extractionRoot = fileManager.temporaryDirectory
            .appendingPathComponent("zipformer-vn-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: extractionRoot, withIntermediateDirectories: true)
        defer {
            try? fileManager.removeItem(at: extractionRoot)
            try? fileManager.removeItem(at: archiveURL)
        }

        try extractTarBz2(archiveURL, to: extractionRoot)

        for fileName in ModelNames.ZipformerVN.requiredModels {
            let destination = targetDir.appendingPathComponent(fileName)
            if force && fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            if !force && fileManager.fileExists(atPath: destination.path) {
                continue
            }

            guard let source = findFile(named: fileName, under: extractionRoot) else {
                throw AsrModelsError.downloadFailed("Release bundle missing \(fileName)")
            }

            try fileManager.copyItem(at: source, to: destination)
        }

        guard modelsExist(at: targetDir) else {
            throw AsrModelsError.downloadFailed("Zipformer VN files were not extracted correctly")
        }

        return targetDir
    }

    public static func downloadAndLoad(to directory: URL? = nil, force: Bool = false) async throws -> ZipformerVnAsrModels {
        let target = try await download(to: directory, force: force)
        return try load(from: target)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        ModelNames.ZipformerVN.requiredModels.allSatisfy { fileName in
            let path = directory.appendingPathComponent(fileName)
            return FileManager.default.fileExists(atPath: path.path)
        }
    }

    public static func defaultCacheDirectory() -> URL {
        ModelCachePaths.modelsRootDirectory().appendingPathComponent(Repo.zipformerVn.folderName, isDirectory: true)
    }

    private static func extractTarBz2(_ archiveURL: URL, to directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xjf", archiveURL.path, "-C", directory.path]

        let errorPipe = Pipe()
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw AsrModelsError.downloadFailed("Failed to extract Zipformer VN archive: \(errorMessage)")
        }
    }

    private static func findFile(named fileName: String, under root: URL) -> URL? {
        let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )

        while let next = enumerator?.nextObject() as? URL {
            guard next.lastPathComponent == fileName else { continue }
            return next
        }

        return nil
    }

    private static func loadTokens(from path: URL) throws -> [Int: String] {
        let text = try String(contentsOf: path, encoding: .utf8)
        var map: [Int: String] = [:]

        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(whereSeparator: \.isWhitespace)
            if parts.count < 2 {
                continue
            }

            guard let id = Int(parts.last!) else {
                continue
            }

            let tokenParts = parts.dropLast()
            let token = tokenParts.joined(separator: " ")
            map[id] = token
        }

        if map.isEmpty {
            throw AsrModelsError.loadingFailed("tokens.txt is empty or invalid")
        }

        logger.info("Loaded \(map.count) Zipformer tokens")
        return map
    }
}

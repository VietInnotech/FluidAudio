import Foundation
import OnnxRuntimeBindings

public struct ZipformerVnAsrModels {
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

        if ORTIsCoreMLExecutionProviderAvailable() {
            let coreMLOptions = ORTCoreMLExecutionProviderOptions()
            coreMLOptions.enableOnSubgraphs = true
            _ = try? sessionOptions.appendCoreMLExecutionProvider(with: coreMLOptions)
        }

        let encoderSession = try ORTSession(env: env, modelPath: encoderPath.path, sessionOptions: sessionOptions)
        let decoderSession = try ORTSession(env: env, modelPath: decoderPath.path, sessionOptions: sessionOptions)
        let joinerSession = try ORTSession(env: env, modelPath: joinerPath.path, sessionOptions: sessionOptions)

        let tokenMap = try loadTokens(from: tokensPath)

        return ZipformerVnAsrModels(
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

        for fileName in ModelNames.ZipformerVN.requiredModels {
            let destination = targetDir.appendingPathComponent(fileName)
            if !force && FileManager.default.fileExists(atPath: destination.path) {
                continue
            }

            let encoded = fileName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? fileName
            let url = try ModelRegistry.resolveModel(Repo.zipformerVn.remotePath, encoded)
            let (data, response) = try await DownloadUtils.fetchWithAuth(from: url)

            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                throw AsrModelsError.downloadFailed("HTTP failure while downloading \(fileName)")
            }

            try data.write(to: destination, options: .atomic)
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

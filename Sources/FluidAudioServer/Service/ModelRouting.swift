import Foundation

/// Supported model identifiers exposed by the server API.
enum ModelID: String, CaseIterable, Sendable {
    case parakeetV2 = "fluidaudio-parakeet-v2"
    case parakeetV3 = "fluidaudio-parakeet-v3"
    case qwen3F32 = "fluidaudio-qwen3-f32"
    case qwen3Int8 = "fluidaudio-qwen3-int8"
    case ctcVi = "fluidaudio-ctc-vi"

    /// Human-readable display name.
    var displayName: String {
        switch self {
        case .parakeetV2: return "Parakeet TDT v2 (0.6B)"
        case .parakeetV3: return "Parakeet TDT v3 (0.6B)"
        case .qwen3F32: return "Qwen3-ASR (0.6B, FP16)"
        case .qwen3Int8: return "Qwen3-ASR (0.6B, Int8)"
        case .ctcVi: return "Parakeet CTC Vietnamese (0.6B)"
        }
    }

    /// Backend engine category for determining which manager to use.
    var backend: ModelBackend {
        switch self {
        case .parakeetV2, .parakeetV3: return .parakeet
        case .qwen3F32, .qwen3Int8: return .qwen3
        case .ctcVi: return .ctc
        }
    }

    /// Whether this model accepts a language parameter.
    var supportsLanguage: Bool {
        switch self {
        case .qwen3F32, .qwen3Int8: return true
        default: return false
        }
    }
}

/// Groupings of model IDs by their underlying engine.
enum ModelBackend: Sendable {
    case parakeet
    case qwen3
    case ctc
}

/// OpenAI-compatible model list response.
struct ModelsListResponse: Codable, Sendable {
    struct ModelObject: Codable, Sendable {
        let id: String
        let object: String
        let created: Int
        let owned_by: String
    }

    let object: String
    let data: [ModelObject]

    static func build() -> ModelsListResponse {
        let now = Int(Date().timeIntervalSince1970)
        let models = ModelID.allCases.map { model in
            ModelObject(
                id: model.rawValue,
                object: "model",
                created: now,
                owned_by: "fluidaudio"
            )
        }
        return ModelsListResponse(object: "list", data: models)
    }
}

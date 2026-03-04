import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

/// Generate OpenAPI 3.1.0 spec as a JSON-encodable dictionary
/// Swift-native implementation with full type safety
func getOpenAPISpec() -> [String: AnyCodableValue] {
    [
        "openapi": "3.1.0",
        "info": [
            "title": "FluidAudio ASR Server API",
            "description":
                "Local speech-to-text server for automatic speech recognition.\n\n## REST API\nOpenAI-compatible `/v1/audio/transcriptions` endpoint for batch transcription.\n\n## WebSocket Streaming API\nReal-time streaming at `ws://host:port/v1/audio/stream`.\n\nSend `{\"type\":\"session.start\",\"model\":\"fluidaudio-parakeet-v3\"}` to begin, binary PCM audio frames (16kHz mono), then `{\"type\":\"session.stop\"}` to finalize.\n\nServer responds with `transcription.partial` (interim) and `transcription.final` messages.",
            "version": "1.1.0",
        ],
        "servers": [
            [
                "url": "http://localhost:8080",
                "description": "Local development server (default)",
            ]
        ],
        "tags": [
            ["name": "Server", "description": "Server health and status"],
            ["name": "Models", "description": "Model management"],
            ["name": "Audio", "description": "Audio transcription"],
        ],
        "paths": [
            "/health": [
                "get": [
                    "tags": ["Server"],
                    "summary": "Health check",
                    "description": "Returns server health status. Requires no authentication.",
                    "operationId": "getHealth",
                    "responses": [
                        "200": [
                            "description": "Server is healthy",
                            "content": [
                                "application/json": [
                                    "schema": [
                                        "type": "object",
                                        "properties": [
                                            "status": ["type": "string", "example": "ok"]
                                        ],
                                    ]
                                ]
                            ],
                        ]
                    ],
                ]
            ],
            "/v1/models": [
                "get": [
                    "tags": ["Models"],
                    "summary": "List available models",
                    "description": "Returns list of available ASR models for transcription.",
                    "operationId": "listModels",
                    "responses": [
                        "200": [
                            "description": "List of available models",
                            "content": [
                                "application/json": [
                                    "schema": [
                                        "type": "object",
                                        "properties": [
                                            "data": [
                                                "type": "array",
                                                "items": [
                                                    "type": "object",
                                                    "properties": [
                                                        "id": ["type": "string"],
                                                        "object": ["type": "string", "example": "model"],
                                                    ],
                                                ],
                                            ]
                                        ],
                                    ]
                                ]
                            ],
                        ]
                    ],
                ]
            ],
            "/v1/audio/transcriptions": [
                "post": [
                    "tags": ["Audio"],
                    "summary": "Transcribe audio",
                    "description":
                        "Transcribe an audio file to text using specified ASR model. OpenAI-compatible multipart/form-data endpoint.",
                    "operationId": "transcribeAudio",
                    "requestBody": [
                        "required": true,
                        "content": [
                            "multipart/form-data": [
                                "schema": [
                                    "type": "object",
                                    "required": ["file", "model"],
                                    "properties": [
                                        "file": [
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Audio file (WAV, MP3, M4A, FLAC)",
                                        ],
                                        "model": [
                                            "type": "string",
                                            "description": "Model ID to use",
                                            "enum": [
                                                "fluidaudio-parakeet-v2",
                                                "fluidaudio-parakeet-v3",
                                                "fluidaudio-ctc-vi",
                                                "fluidaudio-qwen3-f32",
                                                "fluidaudio-qwen3-int8",
                                            ],
                                        ],
                                        "language": [
                                            "type": "string",
                                            "description":
                                                "ISO 639-1 language code (optional)",
                                        ],
                                        "response_format": [
                                            "type": "string",
                                            "enum": ["json", "text", "verbose_json"],
                                            "default": "json",
                                        ],
                                    ],
                                ]
                            ]
                        ],
                    ],
                    "responses": [
                        "200": [
                            "description": "Successful transcription",
                            "content": [
                                "application/json": ["schema": [:]],
                                "text/plain": ["schema": ["type": "string"]],
                            ],
                        ],
                        "400": ["description": "Bad request"],
                        "500": ["description": "Internal server error"],
                    ],
                ]
            ],
        ],
    ]
}

/// Serve Swagger UI with CDN-hosted assets
func getSwaggerUIHTML() -> String {
    #"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="description" content="FluidAudio ASR Server API Documentation" />
        <title>FluidAudio API Docs</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.32.0/swagger-ui.css" />
        <style>
          html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
          *, *:before, *:after { box-sizing: inherit; }
          body { margin: 0; padding: 0; }
        </style>
      </head>
      <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.32.0/swagger-ui-bundle.js" crossorigin></script>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.32.0/swagger-ui-standalone-preset.js" crossorigin></script>
        <script>
          window.onload = function() {
            SwaggerUIBundle({
              url: "/openapi.json",
              dom_id: "#swagger-ui",
              presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIStandalonePreset
              ],
              layout: "StandaloneLayout",
              deepLinking: true,
              persistAuthorization: true,
            });
          };
        </script>
      </body>
    </html>
    """#
}

// MARK: - Codable Value Wrapper

/// Type-erasure wrapper for encoding heterogeneous JSON structures
indirect enum AnyCodableValue: Codable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AnyCodableValue])
    case object([String: AnyCodableValue])

    init(_ value: Any) {
        if value is NSNull {
            self = .null
        } else if let bool = value as? Bool {
            self = .bool(bool)
        } else if let int = value as? Int {
            self = .int(int)
        } else if let double = value as? Double {
            self = .double(double)
        } else if let string = value as? String {
            self = .string(string)
        } else if let array = value as? [Any] {
            self = .array(array.map { AnyCodableValue($0) })
        } else if let dict = value as? [String: Any] {
            self = .object(dict.mapValues { AnyCodableValue($0) })
        } else {
            self = .null
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let b):
            try container.encode(b)
        case .int(let i):
            try container.encode(i)
        case .double(let d):
            try container.encode(d)
        case .string(let s):
            try container.encode(s)
        case .array(let arr):
            try container.encode(arr)
        case .object(let obj):
            try container.encode(obj)
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
        } else if let bool = try? container.decode(Bool.self) {
            self = .bool(bool)
        } else if let int = try? container.decode(Int.self) {
            self = .int(int)
        } else if let double = try? container.decode(Double.self) {
            self = .double(double)
        } else if let string = try? container.decode(String.self) {
            self = .string(string)
        } else if let array = try? container.decode([AnyCodableValue].self) {
            self = .array(array)
        } else if let object = try? container.decode([String: AnyCodableValue].self) {
            self = .object(object)
        } else {
            self = .null
        }
    }
}

// MARK: - Dictionary Literal Support

extension Dictionary where Key == String, Value == AnyCodableValue {
    init(dictionaryLiteral elements: (String, AnyCodableValue)...) {
        self.init()
        for (key, value) in elements {
            self[key] = value
        }
    }
}

extension AnyCodableValue: ExpressibleByDictionaryLiteral {
    init(dictionaryLiteral elements: (String, AnyCodableValue)...) {
        var dict: [String: AnyCodableValue] = [:]
        for (key, value) in elements {
            dict[key] = value
        }
        self = .object(dict)
    }
}

extension AnyCodableValue: ExpressibleByArrayLiteral {
    init(arrayLiteral elements: AnyCodableValue...) {
        self = .array(elements)
    }
}

extension AnyCodableValue: ExpressibleByStringLiteral {
    init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension AnyCodableValue: ExpressibleByIntegerLiteral {
    init(integerLiteral value: Int) {
        self = .int(value)
    }
}

extension AnyCodableValue: ExpressibleByFloatLiteral {
    init(floatLiteral value: Double) {
        self = .double(value)
    }
}

extension AnyCodableValue: ExpressibleByBooleanLiteral {
    init(booleanLiteral value: Bool) {
        self = .bool(value)
    }
}

extension AnyCodableValue: ExpressibleByNilLiteral {
    init(nilLiteral: ()) {
        self = .null
    }
}

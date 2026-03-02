package com.fluidaudio.transcriber

/**
 * Transcription result from the FluidAudio ASR server.
 */
data class TranscriptionResult(
    val text: String,
    val model: String? = null,
    val duration: Double? = null,
    val processingTime: Double? = null
)

/**
 * Model info returned by GET /v1/models.
 */
data class ModelInfo(
    val id: String,
    val objectType: String = "model",
    val ownedBy: String = "fluidaudio"
)

/**
 * Server health response.
 */
data class HealthResponse(
    val status: String
)

/**
 * Connection settings for the FluidAudio server.
 */
data class ServerConfig(
    val baseUrl: String,
    val apiKey: String? = null,
    val model: String = "fluidaudio-parakeet-v3"
) {
    val healthUrl: String get() = "$baseUrl/health"
    val modelsUrl: String get() = "$baseUrl/v1/models"
    val transcriptionsUrl: String get() = "$baseUrl/v1/audio/transcriptions"
}

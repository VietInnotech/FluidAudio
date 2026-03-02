package com.fluidaudio.transcriber

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * HTTP client for the FluidAudio ASR server.
 *
 * All methods are suspend functions that run on [Dispatchers.IO].
 */
class FluidAudioApiClient(private val config: ServerConfig) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)   // transcription can take a while
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    // ── Health ───────────────────────────────────────────────────────────

    /**
     * Check server health. Returns true if the server responds with status "ok".
     */
    suspend fun checkHealth(): Boolean = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url(config.healthUrl)
            .get()
            .build()

        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return@withContext false
                val body = response.body?.string() ?: return@withContext false
                val json = JSONObject(body)
                json.optString("status") == "ok"
            }
        } catch (_: IOException) {
            false
        }
    }

    // ── Models ──────────────────────────────────────────────────────────

    /**
     * Fetch available models from the server.
     */
    suspend fun fetchModels(): List<ModelInfo> = withContext(Dispatchers.IO) {
        val request = newAuthenticatedRequest(config.modelsUrl)
            .get()
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw ApiException(response.code, parseErrorMessage(response.body?.string()))
            }
            val body = response.body?.string() ?: throw ApiException(500, "Empty response")
            val json = JSONObject(body)
            val dataArray = json.optJSONArray("data") ?: JSONArray()
            (0 until dataArray.length()).map { i ->
                val obj = dataArray.getJSONObject(i)
                ModelInfo(
                    id = obj.getString("id"),
                    objectType = obj.optString("object", "model"),
                    ownedBy = obj.optString("owned_by", "fluidaudio")
                )
            }
        }
    }

    // ── Transcription ───────────────────────────────────────────────────

    /**
     * Upload an audio file for transcription.
     *
     * @param audioFile  WAV or other supported audio file.
     * @param language   Optional language hint (e.g., "vi", "en").
     * @param format     Response format: "json", "text", or "verbose_json".
     * @return Parsed [TranscriptionResult].
     * @throws ApiException on HTTP errors (including 429 busy).
     */
    suspend fun transcribe(
        audioFile: File,
        language: String? = null,
        format: String = "verbose_json"
    ): TranscriptionResult = withContext(Dispatchers.IO) {
        val bodyBuilder = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                audioFile.name,
                audioFile.asRequestBody("audio/wav".toMediaType())
            )
            .addFormDataPart("model", config.model)
            .addFormDataPart("response_format", format)

        if (language != null) {
            bodyBuilder.addFormDataPart("language", language)
        }

        val request = newAuthenticatedRequest(config.transcriptionsUrl)
            .post(bodyBuilder.build())
            .build()

        client.newCall(request).execute().use { response ->
            val responseBody = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                throw ApiException(response.code, parseErrorMessage(responseBody))
            }

            when (format) {
                "text" -> TranscriptionResult(text = responseBody.trim())
                "verbose_json" -> {
                    val json = JSONObject(responseBody)
                    TranscriptionResult(
                        text = json.getString("text"),
                        model = json.optString("model").takeIf { it.isNotEmpty() },
                        duration = json.optDouble("duration", Double.NaN).takeIf { !it.isNaN() },
                        processingTime = json.optDouble("processing_time", Double.NaN).takeIf { !it.isNaN() }
                    )
                }
                else -> {
                    val json = JSONObject(responseBody)
                    TranscriptionResult(text = json.getString("text"))
                }
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private fun newAuthenticatedRequest(url: String): Request.Builder {
        val builder = Request.Builder().url(url)
        config.apiKey?.let { key ->
            builder.addHeader("Authorization", "Bearer $key")
        }
        return builder
    }

    private fun parseErrorMessage(body: String?): String {
        if (body.isNullOrBlank()) return "Unknown error"
        return try {
            val json = JSONObject(body)
            json.optJSONObject("error")?.optString("message") ?: body
        } catch (_: Exception) {
            body
        }
    }

    fun shutdown() {
        client.dispatcher.executorService.shutdown()
        client.connectionPool.evictAll()
    }
}

/**
 * Exception representing an HTTP error from the API.
 */
class ApiException(val statusCode: Int, message: String) : IOException("HTTP $statusCode: $message") {
    val isBusy: Boolean get() = statusCode == 429
}

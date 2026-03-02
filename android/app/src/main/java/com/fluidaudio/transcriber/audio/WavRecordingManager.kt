package com.fluidaudio.transcriber.audio

import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Records raw PCM-16 audio into a valid WAV file.
 *
 * Call [start] to begin a new recording, [write] to feed PCM chunks,
 * and [stop] to finalize the WAV header and get the output file.
 */
class WavRecordingManager(private val cacheDir: File) {

    companion object {
        private const val TAG = "WavRecordingManager"
        const val SAMPLE_RATE = 16_000
        const val CHANNELS = 1
        const val BITS_PER_SAMPLE = 16
        private const val HEADER_SIZE = 44
    }

    private var outputStream: FileOutputStream? = null
    private var outputFile: File? = null
    private var totalBytesWritten: Long = 0

    val isRecording: Boolean get() = outputStream != null

    /**
     * Start a new WAV recording. Writes a placeholder header that will be
     * finalized in [stop].
     */
    fun start(): File {
        val file = File(cacheDir, "recording_${System.currentTimeMillis()}.wav")
        outputFile = file
        totalBytesWritten = 0

        val fos = FileOutputStream(file)
        outputStream = fos

        // Write placeholder WAV header (44 bytes)
        fos.write(ByteArray(HEADER_SIZE))

        Log.d(TAG, "Recording started: ${file.absolutePath}")
        return file
    }

    /**
     * Write a chunk of PCM-16 data. Must be called between [start] and [stop].
     */
    fun write(buffer: ByteArray, offset: Int = 0, length: Int = buffer.size) {
        outputStream?.let { fos ->
            fos.write(buffer, offset, length)
            totalBytesWritten += length
        }
    }

    /**
     * Finalize the WAV file: flush the stream and write the correct header.
     *
     * @return The completed WAV file, or null if not recording.
     */
    fun stop(): File? {
        val fos = outputStream ?: return null
        val file = outputFile ?: return null

        try {
            fos.flush()
            fos.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing output stream", e)
        }
        outputStream = null

        // Re-open and overwrite the 44-byte header with correct sizes
        try {
            finalizeWavHeader(file, totalBytesWritten)
        } catch (e: Exception) {
            Log.e(TAG, "Error finalizing WAV header", e)
        }

        Log.d(TAG, "Recording stopped: ${file.absolutePath} (${totalBytesWritten} PCM bytes)")
        return file
    }

    /**
     * Discard the current recording without finalizing.
     */
    fun cancel() {
        try {
            outputStream?.close()
        } catch (_: Exception) {}
        outputStream = null
        outputFile?.delete()
        outputFile = null
        totalBytesWritten = 0
    }

    private fun finalizeWavHeader(file: File, dataSize: Long) {
        val raf = RandomAccessFile(file, "rw")
        raf.seek(0)

        val byteRate = SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE / 8
        val blockAlign = CHANNELS * BITS_PER_SAMPLE / 8

        val header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF chunk
        header.put("RIFF".toByteArray())
        header.putInt((dataSize + HEADER_SIZE - 8).toInt()) // file size - 8
        header.put("WAVE".toByteArray())

        // fmt sub-chunk
        header.put("fmt ".toByteArray())
        header.putInt(16)                    // sub-chunk size
        header.putShort(1)                   // PCM format
        header.putShort(CHANNELS.toShort())
        header.putInt(SAMPLE_RATE)
        header.putInt(byteRate)
        header.putShort(blockAlign.toShort())
        header.putShort(BITS_PER_SAMPLE.toShort())

        // data sub-chunk
        header.put("data".toByteArray())
        header.putInt(dataSize.toInt())

        raf.write(header.array())
        raf.close()
    }
}

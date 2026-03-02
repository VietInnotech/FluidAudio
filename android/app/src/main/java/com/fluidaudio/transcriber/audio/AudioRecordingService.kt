package com.fluidaudio.transcriber.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.FileOutputStream
import java.io.File

/**
 * Foreground service that manages microphone recording.
 *
 * Records 16 kHz mono PCM-16 via [AudioRecord]. Audio data is tee'd to:
 * 1. A [WavRecordingManager] that writes a WAV file.
 * 2. A [ParcelFileDescriptor] pipe feeding [SpeechRecognitionManager] for live captions.
 *
 * Must call [startForeground] with MICROPHONE type BEFORE [AudioRecord.startRecording]
 * on Android 10+.
 */
class AudioRecordingService : Service() {

    companion object {
        private const val TAG = "AudioRecordingService"
        private const val CHANNEL_ID = "audio_recording_channel"
        private const val NOTIFICATION_ID = 1001
        private const val SAMPLE_RATE = 16_000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    // ── Binder ──────────────────────────────────────────────────────────

    inner class LocalBinder : Binder() {
        val service: AudioRecordingService get() = this@AudioRecordingService
    }

    private val binder = LocalBinder()

    override fun onBind(intent: Intent?): IBinder = binder

    // ── State ───────────────────────────────────────────────────────────

    interface Callbacks {
        fun onRecordingStarted()
        fun onRecordingStopped(wavFile: File)
        fun onPartialTranscript(text: String)
        fun onFinalTranscript(text: String)
        fun onRecordingError(message: String)
    }

    private var callbacks: Callbacks? = null
    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private var wavManager: WavRecordingManager? = null
    private var speechManager: SpeechRecognitionManager? = null
    private var pipeWriteFd: ParcelFileDescriptor? = null
    private var pipeWriteStream: FileOutputStream? = null

    @Volatile
    private var isRecording = false

    fun setCallbacks(callbacks: Callbacks?) {
        this.callbacks = callbacks
    }

    fun isCurrentlyRecording(): Boolean = isRecording

    // ── Lifecycle ───────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        wavManager = WavRecordingManager(cacheDir)
        speechManager = SpeechRecognitionManager(this)
        speechManager?.setListener(speechListener)
    }

    override fun onDestroy() {
        stopRecording()
        speechManager?.stop()
        speechManager = null
        wavManager = null
        super.onDestroy()
    }

    // ── Recording ───────────────────────────────────────────────────────

    fun startRecording() {
        if (isRecording) return

        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            callbacks?.onRecordingError("Invalid audio buffer size")
            return
        }

        val recorder = try {
            @Suppress("MissingPermission")
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize * 2
            )
        } catch (e: SecurityException) {
            callbacks?.onRecordingError("Microphone permission not granted")
            return
        }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            callbacks?.onRecordingError("AudioRecord failed to initialize")
            recorder.release()
            return
        }

        audioRecord = recorder

        // Start foreground BEFORE recording (required on Android 10+)
        val notification = buildNotification("Recording…")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        // Set up pipe for SpeechRecognizer
        val pipe = ParcelFileDescriptor.createPipe()
        val pipeRead = pipe[0]
        pipeWriteFd = pipe[1]
        pipeWriteStream = FileOutputStream(pipe[1].fileDescriptor)

        // Start WAV recording
        wavManager?.start()

        // Start speech recognition with pipe read end
        speechManager?.start(pipeRead)

        // Begin audio capture
        isRecording = true
        recorder.startRecording()

        recordingThread = Thread({
            val buffer = ByteArray(bufferSize)
            Log.d(TAG, "Recording thread started")

            while (isRecording) {
                val bytesRead = recorder.read(buffer, 0, buffer.size)
                if (bytesRead > 0) {
                    // Write to WAV file
                    wavManager?.write(buffer, 0, bytesRead)

                    // Tee to SpeechRecognizer pipe
                    try {
                        pipeWriteStream?.write(buffer, 0, bytesRead)
                    } catch (e: Exception) {
                        // Pipe may close if recognizer finishes early — that's OK
                        Log.w(TAG, "Pipe write failed (recognizer may have finished)", e)
                    }
                }
            }

            Log.d(TAG, "Recording thread finished")
        }, "AudioRecordThread").also { it.start() }

        callbacks?.onRecordingStarted()
        Log.d(TAG, "Recording started")
    }

    fun stopRecording() {
        if (!isRecording) return
        isRecording = false

        // Stop and release AudioRecord
        audioRecord?.let { recorder ->
            try {
                recorder.stop()
                recorder.release()
            } catch (e: Exception) {
                Log.w(TAG, "Error stopping AudioRecord", e)
            }
        }
        audioRecord = null

        // Wait for recording thread to finish
        recordingThread?.join(2000)
        recordingThread = null

        // Close pipe
        try {
            pipeWriteStream?.close()
            pipeWriteFd?.close()
        } catch (_: Exception) {}
        pipeWriteStream = null
        pipeWriteFd = null

        // Stop speech recognition
        speechManager?.stop()

        // Finalize WAV
        val wavFile = wavManager?.stop()

        // Stop foreground
        stopForeground(STOP_FOREGROUND_REMOVE)

        if (wavFile != null) {
            callbacks?.onRecordingStopped(wavFile)
            Log.d(TAG, "Recording stopped, WAV file: ${wavFile.absolutePath}")
        }
    }

    // ── SpeechRecognizer callbacks ──────────────────────────────────────

    private val speechListener = object : SpeechRecognitionManager.Listener {
        override fun onPartialResult(text: String) {
            callbacks?.onPartialTranscript(text)
        }

        override fun onFinalResult(text: String) {
            callbacks?.onFinalTranscript(text)
        }

        override fun onError(errorCode: Int) {
            Log.w(TAG, "SpeechRecognizer error: $errorCode")
            // Don't propagate — live captions are best-effort
        }
    }

    // ── Notifications ───────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Audio Recording",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shows while recording audio"
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("FluidAudio")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
    }
}

package com.fluidaudio.transcriber.audio

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.ParcelFileDescriptor
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

/**
 * Wraps Android's [SpeechRecognizer] to provide live caption text from a
 * [ParcelFileDescriptor] pipe (audio source).
 *
 * This gives the user interim transcription text while recording is in
 * progress, before the audio is uploaded to the FluidAudio server.
 */
class SpeechRecognitionManager(private val context: Context) {

    companion object {
        private const val TAG = "SpeechRecognitionMgr"
    }

    interface Listener {
        fun onPartialResult(text: String)
        fun onFinalResult(text: String)
        fun onError(errorCode: Int)
    }

    private var recognizer: SpeechRecognizer? = null
    private var listener: Listener? = null

    val isAvailable: Boolean
        get() = SpeechRecognizer.isRecognitionAvailable(context)

    fun setListener(listener: Listener?) {
        this.listener = listener
    }

    /**
     * Start speech recognition using the read end of a [ParcelFileDescriptor] pipe.
     *
     * @param pipeReadFd The read end of the pipe that carries PCM audio.
     */
    fun start(pipeReadFd: ParcelFileDescriptor) {
        if (!isAvailable) {
            Log.w(TAG, "SpeechRecognizer not available on this device")
            return
        }

        stop()

        recognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
            setRecognitionListener(recognitionListener)
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipeReadFd)
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }

        Log.d(TAG, "Starting speech recognition with pipe input")
        recognizer?.startListening(intent)
    }

    fun stop() {
        recognizer?.let { rec ->
            try {
                rec.stopListening()
                rec.destroy()
            } catch (e: Exception) {
                Log.w(TAG, "Error stopping recognizer", e)
            }
        }
        recognizer = null
    }

    private val recognitionListener = object : RecognitionListener {

        override fun onReadyForSpeech(params: Bundle?) {
            Log.d(TAG, "Ready for speech")
        }

        override fun onBeginningOfSpeech() {
            Log.d(TAG, "Beginning of speech")
        }

        override fun onRmsChanged(rmsdB: Float) {
            // Ignored — we don't display volume level
        }

        override fun onBufferReceived(buffer: ByteArray?) {
            // Ignored
        }

        override fun onEndOfSpeech() {
            Log.d(TAG, "End of speech")
        }

        override fun onError(error: Int) {
            Log.w(TAG, "Recognition error: $error")
            listener?.onError(error)
        }

        override fun onResults(results: Bundle?) {
            val texts = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val text = texts?.firstOrNull() ?: return
            Log.d(TAG, "Final result: $text")
            listener?.onFinalResult(text)
        }

        override fun onPartialResults(partialResults: Bundle?) {
            val texts = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
            val text = texts?.firstOrNull() ?: return
            listener?.onPartialResult(text)
        }

        override fun onEvent(eventType: Int, params: Bundle?) {
            // Ignored
        }
    }
}

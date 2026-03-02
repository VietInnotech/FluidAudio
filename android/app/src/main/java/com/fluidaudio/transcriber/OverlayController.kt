package com.fluidaudio.transcriber

import android.annotation.SuppressLint
import android.content.Context
import android.content.SharedPreferences
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import com.fluidaudio.transcriber.audio.AudioRecordingService
import com.fluidaudio.transcriber.audio.AudioServiceConnection
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File

/**
 * Controls the floating overlay UI: a draggable bubble and an expandable panel.
 *
 * The bubble can be dragged around the screen. Tapping it toggles the panel.
 * The panel has controls for recording, file picking, and displays transcription results.
 */
class OverlayController(private val context: Context) {

    companion object {
        private const val TAG = "OverlayController"
        private const val FILE_POLL_INTERVAL_MS = 500L
    }

    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private val handler = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    // Views
    private var bubbleView: View? = null
    private var panelView: View? = null
    private var isPanelVisible = false

    // Panel UI elements
    private var transcriptTextView: TextView? = null
    private var statusTextView: TextView? = null
    private var progressBar: ProgressBar? = null
    private var scrollView: ScrollView? = null
    private var recordButton: ImageButton? = null
    private var fileButton: ImageButton? = null
    private var closeButton: ImageButton? = null

    // Audio service
    private var audioConnection: AudioServiceConnection? = null

    // API client — loaded from SharedPreferences
    private var apiClient: FluidAudioApiClient? = null
    private var transcriptionJob: Job? = null

    // File picker polling
    private var lastPickTimestamp = 0L
    private val filePickerPrefs: SharedPreferences by lazy {
        context.getSharedPreferences(FilePickerActivity.PREFS_NAME, Context.MODE_PRIVATE)
    }

    // ── Show / Dismiss ──────────────────────────────────────────────────

    fun show() {
        showBubble()
        initApiClient()
        bindAudioService()
    }

    fun dismiss() {
        handler.removeCallbacksAndMessages(null)
        scope.cancel()
        removeBubble()
        removePanel()
        audioConnection?.unbind()
        audioConnection = null
        apiClient?.shutdown()
        apiClient = null
    }

    // ── Bubble ──────────────────────────────────────────────────────────

    @SuppressLint("ClickableViewAccessibility")
    private fun showBubble() {
        val inflater = LayoutInflater.from(context)
        bubbleView = inflater.inflate(R.layout.overlay_bubble, null)

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 50
            y = 200
        }

        bubbleView?.setOnTouchListener(BubbleTouchListener(params))
        windowManager.addView(bubbleView, params)
        Log.d(TAG, "Bubble shown")
    }

    private fun removeBubble() {
        bubbleView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
        }
        bubbleView = null
    }

    // ── Panel ───────────────────────────────────────────────────────────

    private fun showPanel() {
        if (isPanelVisible) return

        val inflater = LayoutInflater.from(context)
        panelView = inflater.inflate(R.layout.overlay_panel, null)

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
        }

        // Bind UI elements
        transcriptTextView = panelView?.findViewById(R.id.transcript_text)
        statusTextView = panelView?.findViewById(R.id.status_text)
        progressBar = panelView?.findViewById(R.id.progress_bar)
        scrollView = panelView?.findViewById(R.id.scroll_view)
        recordButton = panelView?.findViewById(R.id.btn_record)
        fileButton = panelView?.findViewById(R.id.btn_file)
        closeButton = panelView?.findViewById(R.id.btn_close)

        recordButton?.setOnClickListener { toggleRecording() }
        fileButton?.setOnClickListener { pickFile() }
        closeButton?.setOnClickListener { hidePanel() }

        windowManager.addView(panelView, params)
        isPanelVisible = true

        // Start polling for file picker results
        startFilePickerPolling()

        Log.d(TAG, "Panel shown")
    }

    private fun hidePanel() {
        removePanel()
    }

    private fun removePanel() {
        handler.removeCallbacksAndMessages(null)
        panelView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
        }
        panelView = null
        isPanelVisible = false
        transcriptTextView = null
        statusTextView = null
        progressBar = null
        scrollView = null
        recordButton = null
        fileButton = null
        closeButton = null
    }

    private fun togglePanel() {
        if (isPanelVisible) hidePanel() else showPanel()
    }

    // ── Recording ───────────────────────────────────────────────────────

    private fun toggleRecording() {
        val service = audioConnection?.getService()
        if (service == null) {
            setStatus("Audio service not connected")
            return
        }

        if (service.isCurrentlyRecording()) {
            service.stopRecording()
            recordButton?.setImageResource(android.R.drawable.ic_btn_speak_now)
        } else {
            service.startRecording()
            recordButton?.setImageResource(android.R.drawable.ic_media_pause)
        }
    }

    // ── File Picker ─────────────────────────────────────────────────────

    private fun pickFile() {
        FilePickerActivity.launch(context)
    }

    private fun startFilePickerPolling() {
        lastPickTimestamp = filePickerPrefs.getLong(FilePickerActivity.KEY_PICK_TIMESTAMP, 0)
        handler.postDelayed(object : Runnable {
            override fun run() {
                checkForPickedFile()
                if (isPanelVisible) {
                    handler.postDelayed(this, FILE_POLL_INTERVAL_MS)
                }
            }
        }, FILE_POLL_INTERVAL_MS)
    }

    private fun checkForPickedFile() {
        val timestamp = filePickerPrefs.getLong(FilePickerActivity.KEY_PICK_TIMESTAMP, 0)
        if (timestamp > lastPickTimestamp) {
            lastPickTimestamp = timestamp
            val path = filePickerPrefs.getString(FilePickerActivity.KEY_PICKED_FILE_PATH, null) ?: return
            val file = File(path)
            if (file.exists()) {
                Log.d(TAG, "Picked file detected: $path")
                uploadAndTranscribe(file)
            }
        }
    }

    // ── Transcription ───────────────────────────────────────────────────

    private fun uploadAndTranscribe(audioFile: File) {
        val client = apiClient
        if (client == null) {
            setStatus("Server not configured")
            return
        }

        transcriptionJob?.cancel()
        transcriptionJob = scope.launch {
            setStatus("Uploading…")
            showProgress(true)

            try {
                val result = client.transcribe(audioFile)
                showProgress(false)
                setStatus(buildResultStatus(result))
                appendTranscript(result.text)
            } catch (e: ApiException) {
                showProgress(false)
                if (e.isBusy) {
                    setStatus("Server busy — retry in a moment")
                } else {
                    setStatus("Error: ${e.message}")
                }
            } catch (e: Exception) {
                showProgress(false)
                setStatus("Error: ${e.message}")
            }
        }
    }

    private fun buildResultStatus(result: TranscriptionResult): String {
        val parts = mutableListOf<String>()
        result.model?.let { parts.add(it) }
        result.duration?.let { parts.add("${String.format("%.1f", it)}s audio") }
        result.processingTime?.let { parts.add("${String.format("%.2f", it)}s processing") }
        return if (parts.isNotEmpty()) parts.joinToString(" · ") else "Done"
    }

    // ── Audio Service ───────────────────────────────────────────────────

    private fun bindAudioService() {
        audioConnection = AudioServiceConnection(
            context,
            onConnected = { service ->
                service.setCallbacks(audioCallbacks)
                Log.d(TAG, "Audio service connected")
            },
            onDisconnected = {
                Log.d(TAG, "Audio service disconnected")
            }
        )
        audioConnection?.bind()
    }

    private val audioCallbacks = object : AudioRecordingService.Callbacks {
        override fun onRecordingStarted() {
            handler.post { setStatus("Recording…") }
        }

        override fun onRecordingStopped(wavFile: File) {
            handler.post {
                setStatus("Recording stopped")
                recordButton?.setImageResource(android.R.drawable.ic_btn_speak_now)
                uploadAndTranscribe(wavFile)
            }
        }

        override fun onPartialTranscript(text: String) {
            handler.post { setStatus("🎤 $text") }
        }

        override fun onFinalTranscript(text: String) {
            handler.post { setStatus("Live: $text") }
        }

        override fun onRecordingError(message: String) {
            handler.post { setStatus("Error: $message") }
        }
    }

    // ── API Client ──────────────────────────────────────────────────────

    private fun initApiClient() {
        val prefs = context.getSharedPreferences("server_config", Context.MODE_PRIVATE)
        val baseUrl = prefs.getString("base_url", null) ?: return
        val apiKey = prefs.getString("api_key", null)
        val model = prefs.getString("model", "fluidaudio-parakeet-v3") ?: "fluidaudio-parakeet-v3"

        apiClient = FluidAudioApiClient(
            ServerConfig(
                baseUrl = baseUrl,
                apiKey = apiKey,
                model = model
            )
        )
        Log.d(TAG, "API client initialized: $baseUrl, model=$model")
    }

    // ── UI Helpers ──────────────────────────────────────────────────────

    private fun setStatus(text: String) {
        statusTextView?.text = text
    }

    private fun appendTranscript(text: String) {
        val current = transcriptTextView?.text?.toString() ?: ""
        val updated = if (current.isBlank()) text else "$current\n\n$text"
        transcriptTextView?.text = updated
        scrollView?.post { scrollView?.fullScroll(View.FOCUS_DOWN) }
    }

    private fun showProgress(visible: Boolean) {
        progressBar?.visibility = if (visible) View.VISIBLE else View.GONE
    }

    // ── Bubble Touch / Drag ─────────────────────────────────────────────

    @SuppressLint("ClickableViewAccessibility")
    private inner class BubbleTouchListener(
        private val params: WindowManager.LayoutParams
    ) : View.OnTouchListener {

        private var initialX = 0
        private var initialY = 0
        private var touchX = 0f
        private var touchY = 0f
        private var moved = false

        override fun onTouch(v: View, event: MotionEvent): Boolean {
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = params.x
                    initialY = params.y
                    touchX = event.rawX
                    touchY = event.rawY
                    moved = false
                    return true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - touchX).toInt()
                    val dy = (event.rawY - touchY).toInt()
                    if (Math.abs(dx) > 10 || Math.abs(dy) > 10) moved = true
                    params.x = initialX + dx
                    params.y = initialY + dy
                    try {
                        windowManager.updateViewLayout(bubbleView, params)
                    } catch (_: Exception) {}
                    return true
                }
                MotionEvent.ACTION_UP -> {
                    if (!moved) {
                        togglePanel()
                    }
                    return true
                }
            }
            return false
        }
    }
}

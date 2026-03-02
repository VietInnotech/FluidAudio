package com.fluidaudio.transcriber

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/**
 * Main activity for configuring the server connection and managing permissions.
 *
 * This activity is the launcher — the user configures the server URL, API key,
 * and model, grants required permissions, then taps "Start Overlay" to launch
 * the floating transcription panel.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val PERMISSION_REQUEST_CODE = 100
        private const val OVERLAY_PERMISSION_REQUEST_CODE = 101
        private const val PREFS_NAME = "server_config"
    }

    private lateinit var serverUrlInput: EditText
    private lateinit var apiKeyInput: EditText
    private lateinit var modelInput: EditText
    private lateinit var statusText: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var checkHealthButton: Button
    private lateinit var permissionsButton: Button

    private val scope = CoroutineScope(Dispatchers.Main + Job())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        serverUrlInput = findViewById(R.id.input_server_url)
        apiKeyInput = findViewById(R.id.input_api_key)
        modelInput = findViewById(R.id.input_model)
        statusText = findViewById(R.id.text_status)
        startButton = findViewById(R.id.btn_start_overlay)
        stopButton = findViewById(R.id.btn_stop_overlay)
        checkHealthButton = findViewById(R.id.btn_check_health)
        permissionsButton = findViewById(R.id.btn_permissions)

        loadConfig()

        startButton.setOnClickListener { startOverlay() }
        stopButton.setOnClickListener { stopOverlay() }
        checkHealthButton.setOnClickListener { checkHealth() }
        permissionsButton.setOnClickListener { requestAllPermissions() }
    }

    override fun onResume() {
        super.onResume()
        updatePermissionStatus()
    }

    // ── Config ──────────────────────────────────────────────────────────

    private fun loadConfig() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        serverUrlInput.setText(prefs.getString("base_url", "http://192.168.1.42:8080"))
        apiKeyInput.setText(prefs.getString("api_key", ""))
        modelInput.setText(prefs.getString("model", "fluidaudio-parakeet-v3"))
    }

    private fun saveConfig() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit()
            .putString("base_url", serverUrlInput.text.toString().trimEnd('/'))
            .putString("api_key", apiKeyInput.text.toString().ifBlank { null })
            .putString("model", modelInput.text.toString().ifBlank { "fluidaudio-parakeet-v3" })
            .apply()
    }

    // ── Overlay Control ─────────────────────────────────────────────────

    private fun startOverlay() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Overlay permission required", Toast.LENGTH_SHORT).show()
            requestOverlayPermission()
            return
        }

        if (!hasMicrophonePermission()) {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_SHORT).show()
            requestMicrophonePermission()
            return
        }

        saveConfig()
        OverlayService.start(this)
        statusText.text = "Overlay started"
    }

    private fun stopOverlay() {
        OverlayService.stop(this)
        statusText.text = "Overlay stopped"
    }

    // ── Health Check ────────────────────────────────────────────────────

    private fun checkHealth() {
        saveConfig()
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val baseUrl = prefs.getString("base_url", null)

        if (baseUrl.isNullOrBlank()) {
            statusText.text = "Enter a server URL first"
            return
        }

        statusText.text = "Checking…"
        val client = FluidAudioApiClient(
            ServerConfig(
                baseUrl = baseUrl,
                apiKey = prefs.getString("api_key", null)
            )
        )

        scope.launch {
            val healthy = client.checkHealth()
            statusText.text = if (healthy) "✅ Server is healthy" else "❌ Server unreachable"
            client.shutdown()
        }
    }

    // ── Permissions ─────────────────────────────────────────────────────

    private fun requestAllPermissions() {
        if (!Settings.canDrawOverlays(this)) {
            requestOverlayPermission()
            return
        }

        val needed = mutableListOf<String>()
        if (!hasMicrophonePermission()) {
            needed.add(Manifest.permission.RECORD_AUDIO)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                needed.add(Manifest.permission.POST_NOTIFICATIONS)
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_AUDIO)
                != PackageManager.PERMISSION_GRANTED
            ) {
                needed.add(Manifest.permission.READ_MEDIA_AUDIO)
            }
        }

        if (needed.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSION_REQUEST_CODE)
        } else {
            Toast.makeText(this, "All permissions granted", Toast.LENGTH_SHORT).show()
        }
    }

    @Suppress("DEPRECATION")
    private fun requestOverlayPermission() {
        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            Uri.parse("package:$packageName")
        )
        startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE)
    }

    private fun requestMicrophonePermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.RECORD_AUDIO),
            PERMISSION_REQUEST_CODE
        )
    }

    private fun hasMicrophonePermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
    }

    private fun updatePermissionStatus() {
        val items = mutableListOf<String>()

        if (Settings.canDrawOverlays(this)) items.add("✅ Overlay")
        else items.add("❌ Overlay")

        if (hasMicrophonePermission()) items.add("✅ Microphone")
        else items.add("❌ Microphone")

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val notif = ContextCompat.checkSelfPermission(
                this, Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            items.add(if (notif) "✅ Notifications" else "❌ Notifications")
        }

        statusText.text = "Permissions: ${items.joinToString("  ")}"
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        updatePermissionStatus()
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == OVERLAY_PERMISSION_REQUEST_CODE) {
            updatePermissionStatus()
        }
    }
}

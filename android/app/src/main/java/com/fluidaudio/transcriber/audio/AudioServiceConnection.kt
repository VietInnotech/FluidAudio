package com.fluidaudio.transcriber.audio

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log

/**
 * Helper that wraps [ServiceConnection] for binding to [AudioRecordingService].
 */
class AudioServiceConnection(
    private val context: Context,
    private val onConnected: (AudioRecordingService) -> Unit,
    private val onDisconnected: () -> Unit
) {
    companion object {
        private const val TAG = "AudioServiceConn"
    }

    private var service: AudioRecordingService? = null
    private var isBound = false

    val isConnected: Boolean get() = service != null

    fun getService(): AudioRecordingService? = service

    fun bind() {
        if (isBound) return
        val intent = Intent(context, AudioRecordingService::class.java)
        isBound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        Log.d(TAG, "Binding to AudioRecordingService: $isBound")
    }

    fun unbind() {
        if (!isBound) return
        try {
            context.unbindService(connection)
        } catch (e: Exception) {
            Log.w(TAG, "Error unbinding service", e)
        }
        isBound = false
        service = null
    }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as? AudioRecordingService.LocalBinder ?: return
            service = localBinder.service
            Log.d(TAG, "AudioRecordingService connected")
            onConnected(localBinder.service)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
            Log.d(TAG, "AudioRecordingService disconnected")
            onDisconnected()
        }
    }
}

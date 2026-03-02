package com.fluidaudio.transcriber

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Transparent activity for picking audio files.
 *
 * Overlays cannot call [startActivityForResult], so we launch this transparent
 * activity which opens the system file picker, copies the selected file to the
 * app's cache directory, and communicates the result path back via
 * [SharedPreferences] (IPC-safe polling from the overlay service).
 */
class FilePickerActivity : Activity() {

    companion object {
        private const val TAG = "FilePickerActivity"
        private const val PICK_AUDIO_REQUEST = 1

        const val PREFS_NAME = "file_picker_prefs"
        const val KEY_PICKED_FILE_PATH = "picked_file_path"
        const val KEY_PICK_TIMESTAMP = "pick_timestamp"

        fun launch(context: Context) {
            val intent = Intent(context, FilePickerActivity::class.java)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*"
        }

        try {
            startActivityForResult(intent, PICK_AUDIO_REQUEST)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch file picker", e)
            finish()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != PICK_AUDIO_REQUEST) {
            finish()
            return
        }

        if (resultCode != RESULT_OK || data?.data == null) {
            Log.d(TAG, "File picker cancelled")
            finish()
            return
        }

        val uri = data.data!!
        val cachedFile = copyToCache(uri)

        if (cachedFile != null) {
            // Write result to SharedPreferences for the overlay to pick up
            val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            prefs.edit()
                .putString(KEY_PICKED_FILE_PATH, cachedFile.absolutePath)
                .putLong(KEY_PICK_TIMESTAMP, System.currentTimeMillis())
                .apply()

            Log.d(TAG, "File copied to cache: ${cachedFile.absolutePath}")
        } else {
            Log.e(TAG, "Failed to copy file to cache")
        }

        finish()
    }

    private fun copyToCache(uri: Uri): File? {
        return try {
            val inputStream = contentResolver.openInputStream(uri) ?: return null
            val extension = getFileExtension(uri)
            val outputFile = File(cacheDir, "picked_audio_${System.currentTimeMillis()}.$extension")

            FileOutputStream(outputFile).use { output ->
                inputStream.copyTo(output)
            }
            inputStream.close()

            outputFile
        } catch (e: Exception) {
            Log.e(TAG, "Error copying file", e)
            null
        }
    }

    private fun getFileExtension(uri: Uri): String {
        val mimeType = contentResolver.getType(uri)
        return when {
            mimeType?.contains("wav") == true -> "wav"
            mimeType?.contains("mp3") == true || mimeType?.contains("mpeg") == true -> "mp3"
            mimeType?.contains("mp4") == true || mimeType?.contains("m4a") == true -> "m4a"
            mimeType?.contains("flac") == true -> "flac"
            mimeType?.contains("ogg") == true -> "ogg"
            else -> "wav"
        }
    }
}

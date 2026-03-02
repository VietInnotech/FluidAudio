# FluidAudio Android Client — Getting Started

A Kotlin Android app that provides a floating overlay for real-time audio transcription using the FluidAudio ASR server.

## Features

- **Floating overlay** — use transcription from any app via a draggable bubble/panel
- **Microphone recording** — 16 kHz mono PCM with WAV output
- **Live captions** — Android's built-in `SpeechRecognizer` for interim results while recording
- **File upload** — pick audio files from device storage for transcription
- **Server transcription** — uploads recordings to the FluidAudio ASR server for high-quality results

## Prerequisites

1. **FluidAudio ASR server** running on a Mac on the same network  
   (see [Server Getting Started](../ASR/ServerGettingStarted.md))
2. **Android Studio** Hedgehog (2023.1) or later
3. **Android device or emulator** running API 26+ (Android 8.0+)

## Setup

### 1. Start the Server

The server must bind to `0.0.0.0` so the Android device can reach it over LAN:

```bash
FLUIDAUDIO_SERVER_HOST=0.0.0.0 .build/release/fluidaudio-server
```

Note down the Mac's local IP address (e.g., `192.168.1.42`).

### 2. Build the Android App

```bash
cd android
./gradlew assembleDebug
```

### 3. Install on Device

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### 4. Configure

1. Open **FluidAudio Transcriber** on the device
2. Enter the server URL: `http://192.168.1.42:8080`
3. Enter the API key (if the server requires one)
4. Select a model (e.g., `fluidaudio-parakeet-v3`)
5. Grant permissions: Microphone, Overlay ("Draw over other apps"), Notifications

### 5. Use

- Tap **Start Overlay** to launch the floating bubble
- Tap the bubble to expand the transcription panel
- Tap the **mic** button to record → results appear in the panel
- Tap the **file** button to pick an audio file for transcription
- Drag the bubble to reposition it

## Architecture

```
MainActivity
  └─ OverlayService (foreground service)
       ├─ OverlayController (UI: bubble + panel)
       └─ AudioRecordingService (bound foreground service)
            ├─ WavRecordingManager (PCM → WAV file)
            └─ SpeechRecognitionManager (live captions via pipe)

FluidAudioApiClient (OkHttp)
  ├─ GET /health
  ├─ GET /v1/models
  └─ POST /v1/audio/transcriptions
```

## Permissions

| Permission | Purpose |
|---|---|
| `INTERNET` | Communicate with ASR server |
| `RECORD_AUDIO` | Microphone access |
| `FOREGROUND_SERVICE` | Keep services alive |
| `FOREGROUND_SERVICE_MICROPHONE` | Microphone in foreground service |
| `SYSTEM_ALERT_WINDOW` | Floating overlay |
| `POST_NOTIFICATIONS` | Foreground service notification |
| `READ_MEDIA_AUDIO` | File picker access (API 33+) |

## Troubleshooting

- **Server unreachable**: Ensure `FLUIDAUDIO_SERVER_HOST=0.0.0.0` and both devices are on the same network
- **429 Too Many Requests**: Server is busy; the app retries automatically
- **Overlay not showing**: Grant "Draw over other apps" permission in system settings
- **No live captions**: Some devices don't support `SpeechRecognizer` with pipe input; server transcription still works

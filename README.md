# Whisper Transcriber

A Python desktop app for real-time audio transcription on Windows, using OpenAI's Whisper models.

## Features

- **File transcription** — transcribe audio/video files with segment-by-segment progress
- **Live system audio transcription** — capture and transcribe whatever is playing through your speakers (YouTube, Spotify, Telegram calls, etc.) via WASAPI loopback
- **Local-agreement streaming** — tentative text updates in real-time, committed text appears only after 2 consecutive passes agree
- **Model manager** — download, view status, and delete Whisper models (tiny.en, base.en, small.en)

## Prerequisites

- **Python 3.10+**
- **ffmpeg** on PATH (for file transcription of video formats)
- **Windows 10+** (WASAPI loopback requires Windows)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Usage

1. **Select a model** from the dropdown (tiny.en is smallest/fastest)
2. **Download it** via the Model Manager panel on the right
3. **Choose mode:**
   - **File** — pick an audio or video file to transcribe
   - **Live System Audio** — select your output device and click Start to transcribe system audio in real-time
4. **Save** the transcript as a .txt file

## Architecture

| Module | Purpose |
|---|---|
| `main.py` | Entry point |
| `main_window.py` | PySide6 GUI |
| `model_manager.py` | Model download/delete/status |
| `transcriber.py` | faster-whisper wrapper |
| `audio_capture.py` | WASAPI loopback capture |
| `streaming.py` | Local-agreement algorithm (pure Python, testable) |

## Notes

- Models are stored in `%LOCALAPPDATA%\WhisperApp\models`
- Live mode uses a rolling ~8 second audio buffer
- Local-agreement requires 2 consecutive matching passes before text is committed
- Loopback capture needs a WASAPI-compatible audio output device

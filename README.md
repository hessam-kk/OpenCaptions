# Whisper Transcriber

A Python desktop app for real-time audio transcription on Windows, using OpenAI's Whisper models.

## Features

- **File transcription** — transcribe audio/video files with segment-by-segment progress bar
- **Live system audio transcription** — capture and transcribe whatever is playing through your speakers (YouTube, Spotify, Telegram calls, etc.) via WASAPI loopback
- **Microphone transcription** — transcribe live microphone input
- **Local-agreement streaming** — committed text appears only after 2 consecutive inference passes agree, keeping the transcript stable
- **Model manager** — download, view status, and delete Whisper models (tiny.en, base.en, small.en, medium.en) with radio button selection
- **Dark/Light theme** — toggle between dark and light themes
- **Timestamp toggle** — show or hide timestamps in the transcript view (export always includes them)
- **SRT export** — save transcriptions as SRT subtitle files with timestamps
- **Drag and drop** — drag audio/video files onto the window to start transcription
- **Auto-save filenames** — exports default to the same name as the source file

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

1. **Select a model** using the radio buttons in the Models panel on the right
2. **Download it** by clicking the download button (arrow icon) next to the model
3. **Choose mode:**
   - **File** — drag & drop a file or click Browse to select an audio/video file
   - **Live System Audio** — select your output device and click Start to transcribe system audio in real-time
   - **Microphone** — select your microphone device and click Start
4. **Transcribe** — click Start, watch the progress bar, transcript appears in real-time
5. **Export** — use Save Transcript (.txt) or Save SRT (.srt) buttons below the transcript

## Keyboard Shortcuts

- Timestamps can be toggled with the stopwatch button in the save bar
- Clear button resets the transcript and disables save buttons

## Architecture

| Module | Purpose |
|---|---|
| `main.py` | Entry point |
| `main_window.py` | PySide6 GUI (dark/light themes, drag-and-drop, model panel) |
| `model_manager.py` | Model download/delete/status with global HuggingFace cache support |
| `transcriber.py` | faster-whisper wrapper (file + raw buffer modes) |
| `audio_capture.py` | WASAPI loopback and microphone capture via pyaudiowpatch |
| `streaming.py` | Local-agreement algorithm (pure Python, no Qt, testable independently) |

## Notes

- Models are stored in `%LOCALAPPDATA%\WhisperApp\models` and detected from global HuggingFace cache
- Live mode uses a rolling ~5-8 second audio buffer with aggressive trimming to prevent lag
- Local-agreement requires 2 consecutive matching passes before text is committed
- Loopback capture needs a WASAPI-compatible audio output device
- Microphone mode uses standard WASAPI input devices
- All heavy work (model loading, inference, audio capture) runs in background threads to keep the UI responsive

"""WASAPI loopback audio capture via pyaudiowpatch."""

import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from metrics import Metrics

SAMPLING_RATE = 16000
TARGET_CHANNELS = 1


def list_loopback_devices() -> List[Tuple[int, str]]:
    """Return list of (device_index, device_name) for WASAPI loopback devices."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []
    devices = []
    with pyaudio.PyAudio() as p:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return []
        default_speakers_index = wasapi_info.get("defaultOutputDevice")
        if default_speakers_index is None:
            return []
        default_speakers = p.get_device_info_by_index(default_speakers_index)
        for loopback in p.get_loopback_device_info_generator():
            devices.append((loopback["index"], loopback["name"]))
    return devices


def list_microphones() -> List[Tuple[int, str]]:
    """Return list of (device_index, device_name) for microphone/input devices."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []
    devices = []
    with pyaudio.PyAudio() as p:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            return []
        # Get all input devices (not just loopback)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0 and info["hostApi"] == wasapi_info["index"]:
                # Exclude loopback devices (they have "loopback" in name or are output device monitors)
                name = info["name"].lower()
                if "loopback" not in name and "monitor" not in name:
                    devices.append((info["index"], info["name"]))
    return devices


def resample_to_16k(audio: np.ndarray, orig_rate: int) -> np.ndarray:
    """Resample float32 audio from orig_rate to 16kHz using np.interp."""
    if orig_rate == SAMPLING_RATE:
        return audio
    duration = len(audio) / orig_rate
    new_len = int(duration * SAMPLING_RATE)
    x_old = np.linspace(0, duration, len(audio), endpoint=False)
    x_new = np.linspace(0, duration, new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def stereo_to_mono(audio: np.ndarray, channels: int) -> np.ndarray:
    """Mix multi-channel audio down to mono."""
    if channels == 1:
        return audio
    audio = audio.reshape(-1, channels)
    return audio.mean(axis=1).astype(np.float32)


class AudioCapture:
    """Captures system audio via WASAPI loopback."""

    def __init__(self, metrics: Optional[Metrics] = None):
        self._pyaudio = None
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self.metrics = metrics

    def start(self, device_index: int, callback: Callable[[np.ndarray], None]) -> None:
        """Start capturing from a loopback device. Calls callback with float32 16kHz mono chunks."""
        import pyaudiowpatch as pyaudio

        self._callback = callback
        self._running = True
        self._pa = pyaudio
        self._pyaudio = pyaudio.PyAudio()

        device_info = self._pyaudio.get_device_info_by_index(device_index)
        rate = int(device_info["defaultSampleRate"])
        channels = int(device_info["maxInputChannels"])

        self._stream = self._pyaudio.open(
            format=self._pa.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )

        self._thread = threading.Thread(target=self._capture_loop, args=(rate, channels), daemon=True)
        self._thread.start()

    def start_microphone(self, device_index: int, callback: Callable[[np.ndarray], None]) -> None:
        """Start capturing from a microphone/input device. Calls callback with float32 16kHz mono chunks."""
        import pyaudiowpatch as pyaudio

        self._callback = callback
        self._running = True
        self._pa = pyaudio
        self._pyaudio = pyaudio.PyAudio()

        device_info = self._pyaudio.get_device_info_by_index(device_index)
        rate = int(device_info["defaultSampleRate"])
        channels = int(device_info["maxInputChannels"])

        self._stream = self._pyaudio.open(
            format=self._pa.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )

        self._thread = threading.Thread(target=self._capture_loop, args=(rate, channels), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capturing."""
        self._running = False
        # Wait for capture thread to exit
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None

    def _capture_loop(self, rate: int, channels: int) -> None:
        """Background capture loop."""
        while self._running and self._stream:
            try:
                t0 = time.perf_counter()
                data = self._stream.read(1024, exception_on_overflow=False)
                read_sec = time.perf_counter() - t0
            except Exception:
                break  # Stream closed or error — exit loop
            if not self._running:
                break
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            t1 = time.perf_counter()
            audio_np = stereo_to_mono(audio_np, channels)
            audio_np = resample_to_16k(audio_np, rate)
            resample_sec = time.perf_counter() - t1
            if self.metrics:
                self.metrics.record_capture(read_sec, len(audio_np))
                self.metrics.record_resample(resample_sec)
            if self._callback and self._running:
                try:
                    self._callback(audio_np)
                except Exception:
                    break

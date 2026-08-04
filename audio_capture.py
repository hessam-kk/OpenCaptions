"""WASAPI loopback audio capture via pyaudiowpatch."""

import queue
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from metrics import Metrics

SAMPLING_RATE = 16000
TARGET_CHANNELS = 1


class AudioRingBuffer:
    """Thread-safe bounded buffer of float32 mono PCM chunks.

    Capture pushes; the Whisper worker waits for data and takes the oldest
    audio. Overflow (Whisper slower than real-time) drops the oldest audio
    rather than growing unboundedly — capture never blocks on inference.
    """

    MAX_SECONDS = 30.0

    def __init__(self, metrics: Optional[Metrics] = None):
        self.metrics = metrics
        self._q = queue.Queue()
        self._cond = threading.Condition()
        self._pending = 0
        self._dropped = 0
        self._closed = False

    def push(self, chunk: np.ndarray) -> None:
        with self._cond:
            if self._closed:
                return
            self._pending += len(chunk) / SAMPLING_RATE
            while self._pending > self.MAX_SECONDS and not self._q.empty():
                old = self._q.get_nowait()
                self._pending -= len(old) / SAMPLING_RATE
                self._dropped += 1
                if self.metrics:
                    self.metrics.record_ring_drop()
            self._q.put_nowait(chunk)
            self._cond.notify()

    def take(self, max_seconds: float, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Block up to `timeout` for audio; return up to `max_seconds` of the oldest audio."""
        with self._cond:
            if self._pending < 1e-3:
                self._cond.wait(timeout)
            if self._pending < 1e-3:
                return None
            out = []
            budget = max_seconds * SAMPLING_RATE
            while self._pending > 0 and budget > 0 and not self._q.empty():
                chunk = self._q.get_nowait()
                self._pending -= len(chunk) / SAMPLING_RATE
                budget -= len(chunk)
                out.append(chunk)
            return np.concatenate(out).astype(np.float32) if out else None

    def status(self) -> Tuple[float, int]:
        """Return (pending audio seconds, cumulative dropped seconds)."""
        with self._cond:
            return self._pending, self._dropped

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


if __name__ == "__main__":
    # Self-check: push + take round-trips exactly; overflow drops oldest.
    ring = AudioRingBuffer()
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(100):
        chunk = rng.standard_normal(1024).astype(np.float32)
        ring.push(chunk)
        total += len(chunk) / SAMPLING_RATE
    got = ring.take(60.0)
    assert got is not None and abs(len(got) / SAMPLING_RATE - total) < 0.01, "round-trip lost audio"
    big = np.zeros(1024, dtype=np.float32)
    for _ in range(1000):
        ring.push(big)
    pending, dropped = ring.status()
    assert pending <= ring.MAX_SECONDS + 0.01, "buffer exceeded cap"
    assert dropped > 0, "overflow should drop"
    assert ring.take(60.0) is not None
    ring.close()
    print("ring buffer self-check OK")


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

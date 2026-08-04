"""Thread-safe performance counters shared across capture/inference/UI threads."""

import queue
import threading
import time


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.capture_events = 0
        self.capture_duration = 0.0  # seconds spent reading from the audio device
        self.resample_events = 0
        self.resample_duration = 0.0
        self.inference_events = 0
        self.inference_duration = 0.0
        self.audio_received = 0.0  # seconds of audio pushed by capture
        self.audio_processed = 0.0  # seconds of audio covered by completed inference passes
        self.chunk_samples = 0
        self.ring_pending = 0.0  # seconds of audio waiting in the ring buffer
        self.ring_dropped = 0.0  # seconds of audio dropped due to ring overflow
        self.max_ring_pending = 0.0
        self.skipped_drops = 0  # times _drop_old_audio discarded audio
        self._events = queue.Queue(maxsize=2000)
        self._snapshot = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def record_capture(self, duration: float, samples: int) -> None:
        with self._lock:
            self.capture_events += 1
            self.capture_duration += duration
            self.chunk_samples += samples
            self.audio_received += samples / 16000

    def record_resample(self, duration: float) -> None:
        with self._lock:
            self.resample_events += 1
            self.resample_duration += duration

    def record_inference(self, duration: float, processed_sec: float) -> None:
        with self._lock:
            self.inference_events += 1
            self.inference_duration += duration
            self.audio_processed += processed_sec

    def record_ring_drop(self) -> None:
        with self._lock:
            self.ring_dropped += 1

    def set_ring_status(self, pending: float, dropped_sec: float) -> None:
        with self._lock:
            self.ring_pending = pending
            self.max_ring_pending = max(self.max_ring_pending, pending)
            self.ring_dropped = dropped_sec

    def record_drop(self) -> None:
        with self._lock:
            self.skipped_drops += 1

    def log(self, msg: str) -> None:
        try:
            self._events.put_nowait((time.time(), msg))
        except queue.Full:
            pass

    def snapshot(self):
        """Return (capture, resample, inference, ui) seconds for the last window and per-event averages."""
        with self._lock:
            now = time.time()
            last = self._snapshot
            dt = now - last[0]
            if dt <= 0:
                dt = 1e-9
            caps = self.capture_events - last[1]
            res = self.resample_events - last[2]
            infs = self.inference_events - last[3]
            out = (
                now,
                self.capture_events,
                self.resample_events,
                self.inference_events,
                self.capture_duration,
                self.resample_duration,
                self.inference_duration,
                self.audio_received,
                self.audio_processed,
                self.ring_pending,
                self.max_ring_pending,
                self.ring_dropped,
                self.skipped_drops,
            )
            self._snapshot = out
            return {
                "window_sec": dt,
                "capture_per_s": caps / dt,
                "resample_per_s": res / dt,
                "inference_per_s": infs / dt,
                "capture_ms_avg": (self.capture_duration / caps) * 1000 if caps else 0.0,
                "resample_ms_avg": (self.resample_duration / res) * 1000 if res else 0.0,
                "inference_ms_avg": (self.inference_duration / infs) * 1000 if infs else 0.0,
                "capture_sec": self.capture_duration,
                "resample_sec": self.resample_duration,
                "inference_sec": self.inference_duration,
                "inference_events": self.inference_events,
                "audio_received_sec": self.audio_received,
                "audio_processed_sec": self.audio_processed,
                "ring_pending_sec": self.ring_pending,
                "max_ring_pending_sec": self.max_ring_pending,
                "ring_dropped_sec": self.ring_dropped,
                "skipped_drops": self.skipped_drops,
            }

    def drain_log(self, n: int = 200):
        out = []
        while len(out) < n:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                break
        return out

    def format_ts(self, t: float) -> str:
        return time.strftime("%H:%M:%S", time.localtime(t))

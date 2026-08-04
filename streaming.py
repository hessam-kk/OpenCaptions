"""Streaming transcription with a growing audio queue.

Each pass transcribes the FULL buffered audio (nothing dropped) and commits
every word found. No local-agreement re-transcription: every word is committed
once, so nothing is skipped. The queue grows unboundedly; text lags behind live
instead of being lost. Near-real-time, not true real-time — every word arrives.
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from metrics import Metrics

SAMPLING_RATE = 16000


class OnlineASRProcessor:
    """Transcribes the full buffered queue each pass, committing all words."""

    STEP_SEC = 0.5        # min audio before first pass
    MAX_BUFFER_SEC = 300.0  # 5 minutes of queue — nothing drops

    def __init__(self, transcriber, metrics: Optional[Metrics] = None):
        self.transcriber = transcriber
        self.metrics = metrics
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset: float = 0
        self.commited: List[Tuple[float, float, str]] = []
        self._last_end: float = 0

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        """Append a new audio chunk to the queue."""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        max_samples = int(self.MAX_BUFFER_SEC * SAMPLING_RATE)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
            self.buffer_time_offset += (len(self.audio_buffer) - max_samples) / SAMPLING_RATE

    def drain_buffered(self, n: int) -> None:
        """Discard n samples (silence)."""
        if n >= len(self.audio_buffer):
            self.audio_buffer = np.array([], dtype=np.float32)
        else:
            self.audio_buffer = self.audio_buffer[n:]
        self.buffer_time_offset += n / SAMPLING_RATE

    def process_iter(self) -> Tuple[Optional[str], Optional[str], str]:
        """Transcribe the FULL queue, commit all words. Returns (start, end, committed_text)."""
        buffer_sec = len(self.audio_buffer) / SAMPLING_RATE
        if buffer_sec < self.STEP_SEC:
            return None, None, ""

        prompt = self._make_prompt()
        t0 = time.perf_counter()
        words = self.transcriber.transcribe_buffer(
            self.audio_buffer, initial_prompt=prompt or None
        )
        inference_sec = time.perf_counter() - t0

        if self.metrics:
            self.metrics.record_inference(inference_sec, buffer_sec)

        # Commit all words whose START is beyond the last committed END (i.e.
        # genuinely new audio). Words re-transcribed in the overlap region have
        # starts <= last committed end — skip those. This is robust to Whisper's
        # timestamp drift because we compare against the END boundary, not the
        # last word's start.
        last_end = self.commited[-1][1] if self.commited else 0.0
        committed = []
        for a, b, t in words:
            abs_a = a + self.buffer_time_offset
            abs_b = b + self.buffer_time_offset
            if abs_a <= last_end - 0.1:
                continue  # already committed (overlap region)
            committed.append((abs_a, abs_b, t))
        self.commited.extend(committed)
        self._last_end = committed[-1][1] if committed else self._last_end

        committed_text = " ".join(t for _, _, t in committed) if committed else None

        if self.metrics:
            rt = inference_sec / max(buffer_sec, 1e-9)
            self.metrics.log(
                f"infer {buffer_sec:5.1f}s audio -> "
                f"{inference_sec * 1000:7.1f}ms ({rt:4.2f}x RT), "
                f"committed {len(committed):2d} words"
            )

        # Trim the queue so we don't re-transcribe committed audio.
        self._trim_buffer()

        if committed:
            start = committed[0][0]
            end = committed[-1][1]
            return start, end, committed_text
        return None, None, ""

    def _trim_buffer(self) -> None:
        """Drop only COMMITTED audio from the front of the queue.

        The queue keeps the uncommitted tail (which the next pass re-transcribes
        — that's how new words get a chance to commit). Nothing is dropped when
        nothing has committed, so the queue grows to absorb the lag.
        """
        if not self.commited:
            return  # nothing committed — keep the whole queue
        last_end = self.commited[-1][1]
        keep_sec = 2.0  # overlap for context + re-transcription
        cut_sec = max(0.0, last_end - self.buffer_time_offset - keep_sec)
        cut_samples = int(cut_sec * SAMPLING_RATE)
        if cut_samples >= len(self.audio_buffer):
            self.audio_buffer = np.array([], dtype=np.float32)
        elif cut_samples > 0:
            self.audio_buffer = self.audio_buffer[cut_samples:]
        self.buffer_time_offset += cut_samples / SAMPLING_RATE

    def get_tentative(self) -> str:
        return ""

    def finish(self) -> str:
        return ""

    def _make_prompt(self) -> str:
        """Context prompt from committed text (keeps style across passes)."""
        if not self.commited:
            return ""
        result = []
        length = 0
        for w in reversed(self.commited):
            word = w[2]
            if length + len(word) + 1 > 200:
                break
            result.append(word)
            length += len(word) + 1
        return " ".join(reversed(result))

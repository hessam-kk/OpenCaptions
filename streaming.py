"""Local-agreement streaming transcription algorithm.

Pure Python — no Qt dependencies. Testable independently.
Based on the UFAL whisper_streaming approach.
"""

import time
from typing import List, Optional, Tuple

import numpy as np

from metrics import Metrics

SAMPLING_RATE = 16000


class HypothesisBuffer:
    """Manages word-level hypothesis tracking for local agreement."""

    def __init__(self):
        self.commited_in_buffer: List[Tuple[float, float, str]] = []
        self.buffer: List[Tuple[float, float, str]] = []  # previous iteration
        self.new: List[Tuple[float, float, str]] = []  # current iteration
        self.last_commited_time: float = 0
        self.last_commited_word: Optional[str] = None

    def insert(self, new_words: List[Tuple[float, float, str]], offset: float) -> None:
        """Insert new word tuples with time offset applied."""
        new_words = [(a + offset, b + offset, t) for a, b, t in new_words]
        self.new = [(a, b, t) for a, b, t in new_words if a > self.last_commited_time - 0.1]

        if self.new:
            a, _, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                # Deduplicate: remove matching words at the boundary
                cn = len(self.commited_in_buffer)
                nn = len(self.new)
                for i in range(1, min(min(cn, nn), 5) + 1):
                    c = " ".join(self.commited_in_buffer[-j][2] for j in range(1, i + 1)[::-1])
                    tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                    if c == tail:
                        for _ in range(i):
                            self.new.pop(0)
                        break

    def flush(self) -> List[Tuple[float, float, str]]:
        """Return committed words (longest common prefix of last 2 passes)."""
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if not self.buffer:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def complete(self) -> List[Tuple[float, float, str]]:
        """Return all words (committed + tentative)."""
        return self.commited_in_buffer + self.new

    def pop_commited(self, time: float) -> None:
        """Remove committed words that have scrolled past `time`."""
        self.commited_in_buffer = [
            (a, b, t) for a, b, t in self.commited_in_buffer if b > time
        ]


class OnlineASRProcessor:
    """Online ASR processor with streaming chunking.

    Transcribes a head-anchored ~1s window: inference runs on the first 1.0s of
    the buffer (once >= 0.5s has arrived, so passes fire roughly every 0.5s).
    After each pass the buffer is trimmed up to the commit point, so the next
    window overlaps only the uncommitted tail — the local-agreement merge.
    """

    WINDOW_SEC = 1.0      # max audio transcribed per pass (step + overlap)
    STEP_SEC = 0.5        # target cadence between passes

    def __init__(self, transcriber, metrics: Optional[Metrics] = None):
        """
        Args:
            transcriber: A Transcriber instance with .transcribe_buffer()
            metrics: Optional Metrics collector for latency accounting and logging.
        """
        self.transcriber = transcriber
        self.metrics = metrics
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset: float = 0
        self.transcript_buffer = HypothesisBuffer()
        self.commited: List[Tuple[float, float, str]] = []

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        """Append a new audio chunk to the buffer."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def process_iter(self) -> Tuple[Optional[str], Optional[str], str]:
        """Run inference on the head of the buffer and return (start, end, committed_text)
        or (None, None, ''). Also returns tentative text as a second element for display.
        """
        # Wait until at least STEP_SEC of audio has arrived before running
        buffer_sec = len(self.audio_buffer) / SAMPLING_RATE
        if buffer_sec < self.STEP_SEC:
            return None, None, ""

        prompt = self._make_prompt()
        t0 = time.perf_counter()
        words = self.transcriber.transcribe_buffer(
            self.audio_buffer[: int(self.WINDOW_SEC * SAMPLING_RATE)], initial_prompt=prompt
        )
        inference_sec = time.perf_counter() - t0

        if self.metrics:
            self.metrics.record_inference(inference_sec, min(buffer_sec, self.WINDOW_SEC))

        self.transcript_buffer.insert(words, self.buffer_time_offset)

        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)

        committed_text = " ".join(t for _, _, t in committed) if committed else None

        # Tentative = everything not yet committed
        all_words = self.transcript_buffer.complete()
        tentative_text = " ".join(t for _, _, t in all_words) if all_words else ""

        # Trim buffer
        self._trim_buffer()

        if self.metrics:
            rt = inference_sec / max(min(buffer_sec, self.WINDOW_SEC), 1e-9)
            self.metrics.log(
                f"infer {min(buffer_sec, self.WINDOW_SEC):5.1f}s audio -> "
                f"{inference_sec * 1000:7.1f}ms ({rt:4.2f}x RT), "
                f"committed {len(committed):2d} words, "
                f"tentative {len(self.transcript_buffer.new):2d}"
            )

        if committed:
            start = committed[0][0]
            end = committed[-1][1]
            return start, end, committed_text
        return None, None, ""

    def get_tentative(self) -> str:
        """Get current tentative (uncommitted) text."""
        all_words = self.transcript_buffer.complete()
        return " ".join(t for _, _, t in all_words) if all_words else ""

    def finish(self) -> str:
        """Flush remaining tentative text at end of session."""
        remaining = self.transcript_buffer.complete()
        return " ".join(t for _, _, t in remaining) if remaining else ""

    def _make_prompt(self) -> str:
        """Generate a context prompt from committed text that scrolled out."""
        if not self.commited:
            return ""
        k = len(self.commited) - 1
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1
        prompt_words = self.commited[:k]
        # Take up to 200 chars
        result = []
        length = 0
        for w in reversed(prompt_words):
            word = w[2]
            if length + len(word) + 1 > 200:
                break
            result.append(word)
            length += len(word) + 1
        return " ".join(reversed(result))

    def _trim_buffer(self) -> None:
        """Trim the audio buffer: drop committed audio, keep the uncommitted tail.

        The cut point is the last committed word's end time, so the tail that
        remains overlaps the next window — no words are lost at the boundary.
        """
        all_words = self.transcript_buffer.commited_in_buffer
        cut_time = None
        if len(all_words) >= 2:
            cut_time = all_words[-1][1]
        elif all_words:
            cut_time = all_words[0][1]

        if cut_time is not None and cut_time > self.buffer_time_offset:
            cut_seconds = cut_time - self.buffer_time_offset
            samples_to_keep = int(cut_seconds * SAMPLING_RATE)
            if samples_to_keep < len(self.audio_buffer):
                self.audio_buffer = self.audio_buffer[samples_to_keep:]
                self.buffer_time_offset = cut_time
                self.transcript_buffer.pop_commited(cut_time)

        # Hard cap the trailing audio: keep at most WINDOW + STEP of uncommitted audio
        max_samples = int((self.WINDOW_SEC + self.STEP_SEC) * SAMPLING_RATE)
        if len(self.audio_buffer) > max_samples:
            dropped = len(self.audio_buffer) - max_samples
            self.audio_buffer = self.audio_buffer[-max_samples:]
            self.buffer_time_offset += dropped / SAMPLING_RATE
            self.transcript_buffer.pop_commited(self.buffer_time_offset)
            self.commited = [
                (a, b, t) for a, b, t in self.commited
                if b > self.buffer_time_offset
            ]

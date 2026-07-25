"""Whisper transcription wrapper — file and raw buffer modes."""

from typing import Generator, List, Optional, Tuple

import numpy as np
from faster_whisper import WhisperModel
from PySide6.QtCore import QThread, Signal


class Transcriber:
    """Wraps faster-whisper for both file and raw audio buffer transcription."""

    def __init__(self, model_size: str, models_dir: str):
        self.model = WhisperModel(
            model_size, device="cpu", compute_type="int8", download_root=models_dir
        )

    def transcribe_file(
        self, audio_path: str, language: str = "en"
    ) -> Generator[Tuple[int, float, float, str, float], None, None]:
        """Yield (segment_id, start, end, text, total_duration)."""
        segments, info = self.model.transcribe(
            audio_path, language=language, beam_size=1
        )
        duration = info.duration if info.duration else 0
        for i, seg in enumerate(segments):
            yield (i, seg.start, seg.end, seg.text.strip(), duration)

    def transcribe_buffer(
        self,
        audio: np.ndarray,
        language: str = "en",
        initial_prompt: Optional[str] = None,
    ) -> List[Tuple[float, float, str]]:
        """Transcribe a raw float32 16kHz mono buffer. Returns [(start, end, word), ...]."""
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
        )
        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append((w.start, w.end, w.word))
        return words


class FileTranscribeWorker(QThread):
    """Run file transcription in a background thread."""

    segment = Signal(int, float, float, str)  # id, start, end, text
    progress = Signal(float)  # 0.0 - 1.0
    finished = Signal()
    error = Signal(str)

    def __init__(self, transcriber: Transcriber, audio_path: str, language: str = "en"):
        super().__init__()
        self.transcriber = transcriber
        self.audio_path = audio_path
        self.language = language

    def run(self):
        try:
            for seg_id, start, end, text, duration in self.transcriber.transcribe_file(
                self.audio_path, self.language
            ):
                self.segment.emit(seg_id, start, end, text)
                if duration > 0:
                    self.progress.emit(min(end / duration, 1.0))
            self.progress.emit(1.0)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

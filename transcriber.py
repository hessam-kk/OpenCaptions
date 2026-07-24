import os
from faster_whisper import WhisperModel

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


class Transcriber:
    def __init__(self, model_size="tiny.en"):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8",
                                  download_root=_CACHE_DIR)

    def transcribe(self, audio_path, progress_callback=None):
        segments_iter, info = self.model.transcribe(
            audio_path, language="en", beam_size=5
        )
        segments = []
        full_text_parts = []
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())
            if progress_callback:
                progress_callback(seg.end / info.duration if info.duration else 0)
        return segments, " ".join(full_text_parts)

    def generate_srt(self, segments):
        lines = []
        for i, seg in enumerate(segments, 1):
            lines.append(str(i))
            lines.append(f"{self._fmt_ts(seg['start'])} --> {self._fmt_ts(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines)

    def generate_txt(self, segments):
        return "\n".join(seg["text"] for seg in segments)

    @staticmethod
    def _fmt_ts(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

"""Voice Activity Detection — keep silent audio away from Whisper.

Uses an energy-based VAD (RMS per 30ms window) with a relative-noise
threshold, which needs no model download and works everywhere. Silero ONNX
is attempted first; if onnxruntime can't run it, energy VAD is the fallback.
"""

import os
from typing import Optional

import numpy as np

SAMPLING_RATE = 16000
CHUNK = 480  # 30ms windows for energy
SILERO_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


def _default_model_path() -> str:
    return os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
        "OpenCaptionsApp", "vad", "silero_vad.onnx",
    )


class EnergyVAD:
    """Speech gate via windowed RMS energy with adaptive noise floor."""

    def __init__(self, threshold: float = 0.002, noise_floor: float = 0.0003):
        self.threshold = threshold
        self.noise_floor = noise_floor

    def process(self, audio: np.ndarray) -> int:
        # Adaptive noise floor: a window is speech when it's clearly louder
        # than the running noise floor (relative trigger) or above the absolute
        # threshold. The floor adapts to quiet captures (low system volume),
        # so sustained quiet audio isn't permanent silence.
        onset = -1
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            rms = np.sqrt(np.mean(audio[i:i + CHUNK] ** 2))
            if rms > max(self.threshold, self.noise_floor * 4):
                onset = i
                break
            # adapt the floor: drop toward the current rms quickly, rise slowly
            self.noise_floor = max(self.noise_floor * 0.99, rms * 0.5)
        return onset


class SileroVAD:
    """Silero VAD via onnxruntime. Falls back to EnergyVAD if unavailable."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model_path = _default_model_path()
        try:
            import onnxruntime as ort

            if not os.path.exists(self.model_path):
                self._download()
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        except Exception:
            # onnxruntime missing or model incompatible — degrade gracefully
            self.session = None
            self._fallback = EnergyVAD()

    def _download(self):
        import requests

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        r = requests.get(SILERO_URL, timeout=60)
        r.raise_for_status()
        with open(self.model_path, "wb") as f:
            f.write(r.content)

    def process(self, audio: np.ndarray) -> int:
        if self.session is None:
            return self._fallback.process(audio)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        onset = -1
        try:
            for i in range(0, len(audio) - 512 + 1, 512):
                prob, self._state = self.session.run(
                    None,
                    {
                        "input": audio[i:i + 512][None, :],
                        "state": self._state,
                        "sr": np.array(SAMPLING_RATE, dtype=np.int64),
                    },
                )
                if prob[0][0] > self.threshold:
                    onset = i
                    break
        except Exception:
            # Model incompatible with this onnxruntime — degrade once.
            self.session = None
            self._fallback = EnergyVAD()
            return self._fallback.process(audio)
        if onset < 0:
            # Session ran but never flagged speech — likely broken output.
            self.session = None
            self._fallback = EnergyVAD()
            return self._fallback.process(audio)
        return onset


def make_vad() -> SileroVAD:
    """Return a VAD that works: Silero if ORT runs it, else energy fallback."""
    try:
        return SileroVAD()
    except Exception:
        return EnergyVAD()


if __name__ == "__main__":
    vad = make_vad()
    silence = np.zeros(SAMPLING_RATE, dtype=np.float32)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SAMPLING_RATE) * 0.3).astype(np.float32)
    print(f"VAD type: {type(vad).__name__}")
    print(f"1s silence -> onset={vad.process(silence)} (expect -1)")
    print(f"1s noise   -> onset={vad.process(noise)} (expect >=0)")

"""Model download, status, and deletion manager."""

import os
import shutil

from PySide6.QtCore import QThread, Signal

MODELS = {
    "tiny.en": "~75 MB",
    "base.en": "~142 MB",
    "small.en": "~466 MB",
    "medium.en": "~1.5 GB",
}

MODELS_DIR = os.path.join(os.environ.get(
    "LOCALAPPDATA", os.path.expanduser("~")), "AudiscribeApp", "models")


def _all_model_dirs(model_name: str) -> list:
    """Return all possible cache directories for a model."""
    return [
        os.path.join(MODELS_DIR, model_name),  # faster-whisper expected location
        os.path.join(MODELS_DIR, f"models--Systran--faster-whisper-{model_name}"),
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--Systran--faster-whisper-{model_name}"),
    ]


def _find_model_dir(model_name: str):
    """Return the first existing model directory, or None."""
    for d in _all_model_dirs(model_name):
        if os.path.isdir(d):
            return d
    return None


def get_status(model_name: str) -> str:
    """Return 'ready' or 'not_downloaded'."""
    d = _find_model_dir(model_name)
    if d is None:
        return "not_downloaded"
    # Check for model.bin directly in the directory (non-symlink download)
    if os.path.isfile(os.path.join(d, "model.bin")):
        return "ready"
    # Check snapshots dir (HuggingFace cache structure)
    snapshots = os.path.join(d, "snapshots")
    if os.path.isdir(snapshots):
        for snap in os.listdir(snapshots):
            snap_dir = os.path.join(snapshots, snap)
            if os.path.isdir(snap_dir) and any(f.endswith(".bin") for f in os.listdir(snap_dir)):
                return "ready"
    # Check blobs dir
    blobs = os.path.join(d, "blobs")
    if os.path.isdir(blobs) and len(os.listdir(blobs)) > 0:
        return "ready"
    return "not_downloaded"


def get_size(model_name: str) -> int:
    """Return total size in bytes of a downloaded model, or 0."""
    d = _find_model_dir(model_name)
    if d is None:
        return 0
    total = 0
    for dirpath, _, filenames in os.walk(d):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def delete_model(model_name: str) -> None:
    """Delete a model's cache directory (all locations)."""
    for d in _all_model_dirs(model_name):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


class DownloadWorker(QThread):
    """Download a model in a background thread."""

    progress = Signal(int)  # 0-100
    finished = Signal(str)  # model name
    error = Signal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            from huggingface_hub import snapshot_download

            worker = self

            class QtTqdm:
                """Minimal tqdm stand-in for huggingface_hub snapshot downloads in a Qt GUI."""

                def __init__(self, *args, **kwargs):
                    self.total = kwargs.get("total") or 0
                    self.n = kwargs.get("initial", 0) or 0
                    self.desc = kwargs.get("desc", "")
                    self.format_dict = {"rate": None}

                def __enter__(self):
                    return self

                def __exit__(self, *args, **kwargs):
                    pass

                def _emit_progress(self):
                    if self.total and self.total > 0:
                        pct = int(self.n / self.total * 100)
                        worker.progress.emit(min(pct, 100))

                def update(self, n=1):
                    self.n += n
                    self._emit_progress()

                def refresh(self):
                    self._emit_progress()

                def close(self):
                    pass

                def set_description(self, desc=None, refresh=True):
                    if desc is not None:
                        self.desc = desc

                def set_postfix_str(self, postfix="", refresh=False):
                    pass

            os.makedirs(MODELS_DIR, exist_ok=True)
            # Download directly to the location faster-whisper expects: MODELS_DIR/{model_name}/
            snapshot_download(
                repo_id=f"Systran/faster-whisper-{self.model_name}",
                local_dir=os.path.join(MODELS_DIR, self.model_name),
                tqdm_class=QtTqdm,
            )
            self.progress.emit(100)
            self.finished.emit(self.model_name)
        except Exception as e:
            self.error.emit(str(e))


class DeleteWorker(QThread):
    """Delete a model in a background thread."""

    finished = Signal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        delete_model(self.model_name)
        self.finished.emit(self.model_name)

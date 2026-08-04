"""PySide6 main window — ties all modules together."""

import os
import sys
import threading
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QDragEnterEvent, QDropEvent, QTextCursor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QHeaderView

from audio_capture import AudioCapture, AudioRingBuffer, list_loopback_devices, list_microphones
from metrics import Metrics
from model_manager import (
    MODELS,
    MODELS_DIR,
    DownloadWorker,
    DeleteWorker,
    get_size,
    get_status,
)
from streaming import OnlineASRProcessor
from transcriber import FileTranscribeWorker, Transcriber
from vad import make_vad

try:
    import psutil
except ImportError:
    psutil = None

# ── Themes ──────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg": "#1e1e2e", "bg2": "#181825", "surface": "#313244",
        "text": "#cdd6f4", "subtext": "#a6adc8", "muted": "#6c7086",
        "accent": "#89b4fa", "green": "#4ade80", "border": "#45475a",
    },
    "light": {
        "bg": "#eff1f5", "bg2": "#e6e9ef", "surface": "#ccd0da",
        "text": "#4c4f69", "subtext": "#6c6f85", "muted": "#9ca0b0",
        "accent": "#1e66f5", "green": "#40a02b", "border": "#bcc0cc",
    },
}


class WhisperWorker(QThread):
    """Dedicated live-transcription thread: consumes audio from the ring buffer
    and runs inference as fast as data arrives, never blocking capture."""

    result = Signal(object)  # (start, end, committed_text, tentative_text)

    MAX_BATCH_SECONDS = 12.0

    def __init__(self, transcriber, ring: AudioRingBuffer, metrics: Metrics):
        super().__init__()
        self.transcriber = transcriber
        self.ring = ring
        self.metrics = metrics
        self.processor = OnlineASRProcessor(transcriber, metrics=metrics)
        self.vad: Optional[SileroVAD] = None
        self._stop_flag = threading.Event()

    def stop(self):
        self._stop_flag.set()
        self.ring.close()

    def run(self):
        def _log_unhandled(exc_type, exc, tb):
            self.metrics.log(f"LIVE WORKER CRASH: {exc}")
        sys.excepthook = _log_unhandled
        try:
            # Near-real-time: pull audio into the processor's queue as fast as
            # it arrives, and transcribe the whole queue each pass. Nothing is
            # dropped — the queue absorbs the lag (text appears late, not lost).
            while not self._stop_flag.is_set():
                # Pull a batch of audio (accumulate until >= STEP_SEC so the
                # VAD has a real window; capture delivers tiny 0.02s chunks).
                batch = None
                while not self._stop_flag.is_set():
                    audio = self.ring.take(self.MAX_BATCH_SECONDS, timeout=0.5)
                    if audio is None:
                        continue
                    if batch is None:
                        batch = audio
                    else:
                        batch = np.concatenate([batch, audio])
                    if len(batch) / 16000 >= self.processor.STEP_SEC:
                        break
                if self._stop_flag.is_set() or batch is None:
                    continue
                if self.metrics:
                    pending, dropped = self.ring.status()
                    self.metrics.set_ring_status(pending, dropped)
                rms = float(np.sqrt(np.mean(batch ** 2)))
                self.metrics.log(f"[worker] took {len(batch)/16000:.2f}s batch, rms={rms:.4f}")
                self._maybe_init_vad()
                if self.vad is not None:
                    onset = self.vad.process(batch)
                    self.metrics.log(f"[worker] VAD onset={onset} (rms={rms:.4f})")
                    if onset < 0:
                        # Silence: drop it without touching Whisper.
                        if self.metrics:
                            self.metrics.record_silence(len(batch) / 16000)
                        continue
                self.processor.insert_audio_chunk(batch)
                # Run one pass on the whole queue, commit everything.
                try:
                    start, end, committed = self.processor.process_iter()
                    self.metrics.log(f"[worker] pass done: committed={committed!r} buf={len(self.processor.audio_buffer)/16000:.2f}s")
                except Exception as e:
                    self.metrics.log(f"live worker error: {e}")
                    continue
                tentative = self.processor.get_tentative()
                self.metrics.log(f"[worker] emitting result: committed={committed!r} tentative={tentative!r}")
                self.result.emit((start, end, committed or "", tentative))
        finally:
            self.processor = None

    def _maybe_init_vad(self):
        if self.vad is None:
            try:
                self.vad = make_vad()
                self.metrics.log(f"VAD loaded: {type(self.vad).__name__}")
            except Exception as e:
                self.metrics.log(f"VAD unavailable ({e}); running without VAD")
                self.vad = None  # avoid retrying every pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCaptions")
        self.setMinimumSize(900, 600)

        self._transcriber: Optional[Transcriber] = None
        self._file_worker: Optional[FileTranscribeWorker] = None
        self.console = None
        self.metrics = Metrics()
        self._capture = AudioCapture(self.metrics)
        self._ring: Optional[AudioRingBuffer] = None
        self._live_worker: Optional[WhisperWorker] = None
        self._status_timer = QTimer()
        self._status_timer.timeout.connect(self._update_status_metrics)
        self._status_timer.start(1000)
        self._selected_file: Optional[str] = None
        self._selected_model: str = list(MODELS.keys())[0]
        self._theme: str = "dark"
        self._segments: list = []  # [(start, end, text), ...] for SRT export
        self._show_timestamps: bool = True

        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_theme()
        self._refresh_model_status()
        self._on_mode_changed(0)

    # ── UI Construction ─────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(6)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)
        toolbar.addWidget(QLabel("Mode:"))
        self.mode_file_radio = QRadioButton("File")
        self.mode_file_radio.setChecked(True)
        self.mode_live_radio = QRadioButton("Live System Audio (beta)")
        self.mode_mic_radio = QRadioButton("Microphone")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_file_radio, 0)
        self._mode_group.addButton(self.mode_live_radio, 1)
        self._mode_group.addButton(self.mode_mic_radio, 2)
        self._mode_group.setExclusive(True)
        self._mode_group.idClicked.connect(self._on_mode_changed)
        toolbar.addWidget(self.mode_file_radio)
        toolbar.addWidget(self.mode_live_radio)
        toolbar.addWidget(self.mode_mic_radio)

        toolbar.addStretch()

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedHeight(32)
        self.start_btn.setStyleSheet("QPushButton { padding: 6px 20px; font-weight: bold; }")
        self.start_btn.clicked.connect(self._on_start_stop)
        toolbar.addWidget(self.start_btn)

        self.debug_btn = QPushButton("\U0001f4ca")  # 📊
        self.debug_btn.setFixedSize(32, 32)
        self.debug_btn.setToolTip("Toggle debug console")
        self.debug_btn.setCheckable(True)
        self.debug_btn.clicked.connect(self._toggle_console)
        toolbar.addWidget(self.debug_btn)

        self.theme_btn = QPushButton("\u263e")  # ☾
        self.theme_btn.setFixedSize(32, 32)
        self.theme_btn.setToolTip("Toggle dark/light theme")
        self.theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(self.theme_btn)

        layout.addLayout(toolbar)

        # Unified selector bar (file or device depending on mode)
        self.selector_bar = QWidget()
        selector_layout = QHBoxLayout(self.selector_bar)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(6)

        self.selector_label = QLabel("File:")
        selector_layout.addWidget(self.selector_label)

        # File picker elements
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected — drag & drop or click Browse")
        self.file_path_edit.setReadOnly(True)
        selector_layout.addWidget(self.file_path_edit)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedHeight(32)
        self.browse_btn.clicked.connect(self._pick_file)
        selector_layout.addWidget(self.browse_btn)

        # Device picker elements
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("deviceCombo")
        self._populate_devices()
        selector_layout.addWidget(self.device_combo, 1)  # stretch=1 for full width

        layout.addWidget(self.selector_bar)

        # Main splitter: transcript + model panel
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setFont(QFont("Cascadia Code", 10))
        splitter.addWidget(self.transcript)

        # Right: model manager panel with radio buttons
        model_panel = QWidget()
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(4)
        model_panel.setMaximumWidth(280)
        model_panel.setMinimumWidth(220)

        header = QLabel("Models")
        header.setStyleSheet("font-weight: bold; font-size: 13px; color: #cdd6f4;")
        model_layout.addWidget(header)

        # Radio button group for model selection
        self._model_group = QButtonGroup(self)
        self._model_group.setExclusive(True)
        self.model_widgets = {}

        for i, (name, size_str) in enumerate(MODELS.items()):
            group = QGroupBox()
            g_layout = QVBoxLayout(group)
            g_layout.setContentsMargins(8, 6, 8, 6)
            g_layout.setSpacing(4)

            # Top row: radio + name + status dot
            top_row = QHBoxLayout()
            radio = QRadioButton(f"{name}")
            radio.setChecked(i == 0)
            radio.setStyleSheet("font-weight: bold; font-size: 12px;")
            self._model_group.addButton(radio, i)
            top_row.addWidget(radio)

            self._model_group.idToggled.connect(self._on_model_radio_changed)

            status_dot = QLabel()
            status_dot.setFixedSize(10, 10)
            status_dot.setStyleSheet("background-color: #6c7086; border-radius: 5px;")
            top_row.addWidget(status_dot)
            top_row.addStretch()

            size_label = QLabel(f"{size_str}")
            size_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
            top_row.addWidget(size_label)

            g_layout.addLayout(top_row)

            # Status text + disk size
            status_label = QLabel(f"Not downloaded")
            status_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
            g_layout.addWidget(status_label)

            disk_label = QLabel("")
            disk_label.setStyleSheet("color: #6c7086; font-size: 10px;")
            g_layout.addWidget(disk_label)

            # Action button (toggle between download/delete)
            action_btn = QPushButton()
            action_btn.setFixedHeight(28)
            action_btn.setStyleSheet("font-size: 11px; padding: 4px 12px; margin-bottom: 4px;")
            action_btn.clicked.connect(lambda checked, n=name: self._on_model_action(n))
            g_layout.addWidget(action_btn, 0, Qt.AlignRight)

            progress = QProgressBar()
            progress.setFixedHeight(4)
            progress.setVisible(False)
            progress.setStyleSheet(
                "QProgressBar { background-color: #313244; border: none; }"
                "QProgressBar::chunk { background-color: #89b4fa; }"
            )
            g_layout.addWidget(progress)

            self.model_widgets[name] = {
                "group": group,
                "radio": radio,
                "dot": status_dot,
                "status": status_label,
                "disk": disk_label,
                "action_btn": action_btn,
                "progress": progress,
            }
            model_layout.addWidget(group)

            # Store first model as selected
            if i == 0:
                self._selected_model = name

        model_layout.addStretch()
        splitter.addWidget(model_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)  # stretch=1 so it fills vertical space

        # Save bar below transcript
        save_bar = QHBoxLayout()
        save_bar.setSpacing(8)
        self.save_btn = QPushButton("Save Transcript (.txt)")
        self.save_btn.setFixedHeight(30)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_transcript)
        save_bar.addWidget(self.save_btn)
        self.save_srt_btn = QPushButton("Save SRT (.srt)")
        self.save_srt_btn.setFixedHeight(30)
        self.save_srt_btn.setEnabled(False)
        self.save_srt_btn.clicked.connect(self._save_srt)
        save_bar.addWidget(self.save_srt_btn)
        save_bar.addStretch()
        self.ts_toggle_btn = QPushButton("\u23f1 Timestamps")  # ⏱
        self.ts_toggle_btn.setFixedHeight(30)
        self.ts_toggle_btn.setToolTip("Toggle timestamps in transcript")
        self.ts_toggle_btn.setCheckable(True)
        self.ts_toggle_btn.setChecked(True)
        self.ts_toggle_btn.clicked.connect(self._toggle_timestamps)
        save_bar.addWidget(self.ts_toggle_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(30)
        self.clear_btn.clicked.connect(self._clear_transcript)
        save_bar.addWidget(self.clear_btn)
        layout.addLayout(save_bar)

        # Transcription progress bar
        self.trans_progress = QProgressBar()
        self.trans_progress.setVisible(False)
        self.trans_progress.setFixedHeight(6)
        self.trans_progress.setStyleSheet(
            "QProgressBar { background-color: #313244; border: none; }"
            "QProgressBar::chunk { background-color: #89b4fa; }"
        )
        layout.addWidget(self.trans_progress)

        self.trans_progress_label = QLabel("")
        self.trans_progress_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self.trans_progress_label)

        # Metrics bar (latency / queue / CPU / rate)
        self.metrics_bar = QHBoxLayout()
        self.metrics_bar.setSpacing(16)
        self.metrics_bar.addStretch()
        self.latency_label = QLabel("behind —")
        self.latency_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-family: Consolas;")
        self.metrics_bar.addWidget(self.latency_label)
        self.queue_label = QLabel("queue —")
        self.queue_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-family: Consolas;")
        self.metrics_bar.addWidget(self.queue_label)
        self.cpu_label = QLabel("cpu —")
        self.cpu_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-family: Consolas;")
        self.metrics_bar.addWidget(self.cpu_label)
        self.rate_label = QLabel("rate —")
        self.rate_label.setStyleSheet("color: #a6adc8; font-size: 11px; font-family: Consolas;")
        self.metrics_bar.addWidget(self.rate_label)
        layout.addLayout(self.metrics_bar)

        # Status bar
        self.statusBar().showMessage("Idle")
        self.statusBar().setStyleSheet("QStatusBar { color: #a6adc8; }")
        self._credit_label = QLabel('Made with \u2764 by <a href="https://github.com/hessam-kk/OpenCaptions" style="color:#a6adc8;">Hessam_kk</a>')
        self._credit_label.setOpenExternalLinks(True)
        self._credit_label.setStyleSheet("font-size: 10px; color: #a6adc8; padding-right: 4px;")
        self.statusBar().addPermanentWidget(self._credit_label)

    # ── Debug console ───────────────────────────────────────────────────
    def _toggle_console(self):
        if self.console is None:
            self.console = DebugConsole(self.metrics)
        if self.console.isVisible():
            self.console.hide()
        else:
            self.console.show()

    # ── Theme ────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        self._theme = "light" if self._theme == "dark" else "dark"
        self._apply_theme()

    def _apply_theme(self):
        t = THEMES[self._theme]
        is_dark = self._theme == "dark"
        self.theme_btn.setText("\u2600" if is_dark else "\u263e")  # ☀ or ☾

        # Window + central widget
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background-color: {t['bg']}; color: {t['text']}; }}
            QGroupBox {{ border: 1px solid {t['border']}; border-radius: 4px; margin-top: 4px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {t['subtext']}; }}
            QLabel {{ color: {t['text']}; }}
            QRadioButton {{ color: {t['text']}; spacing: 6px; }}
            QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 7px; border: 2px solid {t['border']}; background: {t['bg2']}; }}
            QRadioButton::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
            QComboBox {{ background-color: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 4px 8px; border-radius: 4px; }}
            #deviceCombo:hover {{ border-color: {t['accent']}; background-color: {t['bg2']}; }}
            #deviceCombo:on {{ background-color: {t['bg2']}; border-color: {t['accent']}; }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{ background-color: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; selection-background-color: {t['accent']}; selection-color: {t['bg']}; padding: 4px; outline: none; }}
            QComboBox QAbstractItemView::item {{ padding: 6px 8px; min-height: 24px; }}
            QComboBox QAbstractItemView::item:hover {{ background-color: {t['accent']}; color: {t['bg']}; }}
            QComboBox QAbstractItemView::item:selected {{ background-color: {t['accent']}; color: {t['bg']}; }}
            #deviceCombo:hover {{ border-color: {t['accent']}; background-color: {t['surface']}; }}
            #deviceCombo:on {{ background-color: {t['bg2']}; border-color: {t['accent']}; }}
            QLineEdit {{ background-color: {t['bg2']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px; border-radius: 4px; }}
            QPushButton {{ background-color: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 4px 12px; }}
            QPushButton:hover {{ background-color: {t['accent']}; color: {t['bg']}; border-color: {t['accent']}; }}
            QPushButton:disabled {{ color: {t['muted']}; border-color: {t['bg2']}; }}
            QTextEdit {{ background-color: {t['bg2']}; color: {t['text']}; border: 1px solid {t['border']}; }}
            QLineEdit {{ background-color: {t['bg2']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 6px; border-radius: 4px; }}
            QProgressBar {{ background-color: {t['surface']}; border: none; }}
            QProgressBar::chunk {{ background-color: {t['accent']}; }}
            QStatusBar {{ color: {t['subtext']}; }}
            QSplitter::handle {{ background-color: {t['border']}; }}
        """)

        # Dynamic widget colors that can't be set via stylesheet
        for name, widgets in self.model_widgets.items():
            status = get_status(name)
            is_ready = status == "ready"
            widgets["status"].setStyleSheet(f"color: {t['green'] if is_ready else t['subtext']}; font-size: 11px;")
            widgets["dot"].setStyleSheet(f"background-color: {t['green'] if is_ready else t['muted']}; border-radius: 5px;")
            widgets["disk"].setStyleSheet(f"color: {t['muted']}; font-size: 10px;")

    def _toggle_timestamps(self):
        self._show_timestamps = self.ts_toggle_btn.isChecked()
        self._refresh_transcript()

    def _refresh_transcript(self):
        """Re-render the transcript with or without timestamps."""
        if not self._segments:
            return
        t = THEMES[self._theme]
        self.transcript.clear()
        for start, end, text in self._segments:
            if self._show_timestamps:
                ts = f"[{self._fmt_ts(start)} → {self._fmt_ts(end)}]"
                self.transcript.append(f'<span style="color:{t["accent"]};">{ts}</span> {text}')
            else:
                self.transcript.append(text)

    def _clear_transcript(self):
        """Clear the transcript and segments, disable save buttons."""
        self.transcript.clear()
        self._segments = []
        self.save_btn.setEnabled(False)
        self.save_srt_btn.setEnabled(False)

    # ── Radio button handler ────────────────────────────────────────────
    @Slot(int, bool)
    def _on_model_radio_changed(self, id: int, checked: bool):
        if checked:
            for name, widgets in self.model_widgets.items():
                if widgets["radio"].isChecked():
                    self._selected_model = name
                    break

    # ── Device listing ──────────────────────────────────────────────────
    def _populate_devices(self):
        self.device_combo.clear()
        devices = list_loopback_devices()
        if not devices:
            self.device_combo.addItem("(no loopback devices found)")
            self.device_combo.setEnabled(False)
        else:
            for idx, name in devices:
                self.device_combo.addItem(name, idx)

    def _populate_microphones(self):
        self.device_combo.clear()
        devices = list_microphones()
        if not devices:
            self.device_combo.addItem("(no microphones found)")
            self.device_combo.setEnabled(False)
        else:
            for idx, name in devices:
                self.device_combo.addItem(name, idx)

    # ── Model management ────────────────────────────────────────────────
    def _refresh_model_status(self):
        for name, widgets in self.model_widgets.items():
            status = get_status(name)
            size = get_size(name)
            is_ready = status == "ready"

            widgets["status"].setText("Ready" if is_ready else "Not downloaded")
            widgets["status"].setStyleSheet(
                f"color: {'#4ade80' if is_ready else '#a6adc8'}; font-size: 11px;"
            )
            widgets["dot"].setStyleSheet(
                f"background-color: {'#4ade80' if is_ready else '#6c7086'}; border-radius: 5px;"
            )
            if size > 0:
                widgets["disk"].setText(f"{size // (1024*1024)} MB on disk")
            else:
                widgets["disk"].setText("")
            # Unified action button: arrow-down to download, trash to delete
            if is_ready:
                widgets["action_btn"].setText("\U0001f5d1 Delete")  # 🗑 Delete
                widgets["action_btn"].setToolTip("Delete model")
            else:
                widgets["action_btn"].setText("\u2b07 Download")  # ⬇ Download
                widgets["action_btn"].setToolTip("Download model")
            widgets["progress"].setVisible(False)

    def _on_model_action(self, name: str):
        """Toggle between download and delete based on current status."""
        if get_status(name) == "ready":
            self._delete_model(name)
        else:
            self._download_model(name)

    def _download_model(self, name: str):
        widgets = self.model_widgets[name]
        widgets["action_btn"].setEnabled(False)
        widgets["progress"].setVisible(True)
        widgets["progress"].setValue(0)
        self.statusBar().showMessage(f"Downloading model {name}...")
        self._dl_worker = DownloadWorker(name)
        self._dl_worker.progress.connect(lambda p, w=widgets: w["progress"].setValue(p))
        self._dl_worker.finished.connect(self._on_download_done)
        self._dl_worker.error.connect(self._on_download_error)
        self._dl_worker.start()

    @Slot(str)
    def _on_download_done(self, name: str):
        self.statusBar().showMessage(f"Model {name} ready")
        self._refresh_model_status()

    @Slot(str)
    def _on_download_error(self, msg: str):
        self.statusBar().showMessage("Download failed")
        QMessageBox.critical(self, "Download Error", msg)
        self._refresh_model_status()

    def _delete_model(self, name: str):
        reply = QMessageBox.question(
            self, "Delete Model", f"Delete {name} model from disk?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._del_worker = DeleteWorker(name)
            self._del_worker.finished.connect(lambda: self._refresh_model_status())
            self._del_worker.start()

    # ── Mode switching ──────────────────────────────────────────────────
    @Slot(int)
    def _on_mode_changed(self, index: int):
        is_file = index == 0
        is_live = index == 1
        is_mic = index == 2
        self.selector_label.setText("File:" if is_file else "Device:")
        self.file_path_edit.setVisible(is_file)
        self.browse_btn.setVisible(is_file)
        self.device_combo.setVisible(is_live or is_mic)
        if is_mic:
            self._populate_microphones()
        else:
            self._populate_devices()

    # ── Start / Stop ────────────────────────────────────────────────────
    def _on_start_stop(self):
        if self.start_btn.text() == "Start":
            self._start()
        else:
            self._stop()

    def _start(self):
        model_name = self._selected_model
        if get_status(model_name) != "ready":
            QMessageBox.warning(self, "Model not ready", f"Please download {model_name} first.")
            return

        self.statusBar().showMessage("Loading model...")
        try:
            self._transcriber = Transcriber(model_name, MODELS_DIR)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            self.statusBar().showMessage("Idle")
            return

        self.start_btn.setText("Stop")
        self.save_btn.setEnabled(True)
        self.save_srt_btn.setEnabled(True)
        self._segments = []
        self.transcript.clear()

        if self._mode_group.checkedId() == 0:
            self._start_file_mode()
        else:
            self._start_live_mode()

    def _stop(self):
        if self.metrics:
            self.metrics.log("capture stopped")
        # Stop audio capture FIRST to prevent callbacks after ring is closed
        try:
            self._capture.stop()
        except Exception:
            pass
        # Close the ring so the worker's take() wakes and exits
        if self._ring:
            self._ring.close()
        # Stop the live worker - wait for it to finish its current pass
        if self._live_worker:
            try:
                self._live_worker.result.disconnect()
            except RuntimeError:
                pass
            try:
                self._live_worker.finished.disconnect()
            except RuntimeError:
                pass
            self._live_worker.stop()
            if self._live_worker.isRunning():
                self._live_worker.wait(5000)
            self._live_worker = None
        self._ring = None
        # Stop file worker
        if self._file_worker:
            if self._file_worker.isRunning():
                self._file_worker.terminate()
                self._file_worker.wait(2000)
            self._file_worker = None
        self.start_btn.setText("Start")
        has_content = bool(self._segments)
        self.save_btn.setEnabled(has_content)
        self.save_srt_btn.setEnabled(has_content)
        self.trans_progress.setVisible(False)
        self.trans_progress_label.setText("")
        self.statusBar().showMessage("Idle")

    # ── File selection ──────────────────────────────────────────────────
    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio/video file", "",
            "Audio/Video (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.mp4 *.mkv *.avi *.mov *.webm);;All files (*.*)"
        )
        if path:
            self._selected_file = path
            self.file_path_edit.setText(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            supported = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
                         ".aac", ".mp4", ".mkv", ".avi", ".mov", ".webm"}
            if ext in supported:
                self._selected_file = path
                self.file_path_edit.setText(path)
            else:
                QMessageBox.warning(self, "Unsupported format",
                                    f"Cannot transcribe {ext} files.\n"
                                    f"Supported: MP3, WAV, M4A, FLAC, OGG, MP4, MKV, etc.")

    # ── File mode ───────────────────────────────────────────────────────
    def _start_file_mode(self):
        path = self._selected_file
        if not path:
            self._stop()
            return

        self.statusBar().showMessage("Transcribing...")
        self.trans_progress.setValue(0)
        self.trans_progress.setVisible(True)
        self.trans_progress_label.setText("Transcribing...")
        self._file_worker = FileTranscribeWorker(self._transcriber, path)
        self._file_worker.segment.connect(self._on_file_segment)
        self._file_worker.progress.connect(self._on_file_progress)
        self._file_worker.finished.connect(self._on_file_done)
        self._file_worker.error.connect(self._on_file_error)
        self._file_worker.start()

    @Slot(int, float, float, str)
    def _on_file_segment(self, seg_id: float, start: float, end: float, text: str):
        self._segments.append((start, end, text))
        t = THEMES[self._theme]
        if self._show_timestamps:
            ts = f"[{self._fmt_ts(start)} → {self._fmt_ts(end)}]"
            self.transcript.append(f'<span style="color:{t["accent"]};">{ts}</span> {text}')
        else:
            self.transcript.append(text)

    @Slot(float)
    def _on_file_progress(self, pct: float):
        self.trans_progress.setValue(int(pct * 100))
        self.trans_progress_label.setText(f"Transcribing... {int(pct * 100)}%")

    @Slot()
    def _on_file_done(self):
        self.trans_progress.setValue(100)
        self.trans_progress_label.setText("Complete")
        self.statusBar().showMessage("Transcription complete")
        self._file_worker = None
        self.start_btn.setText("Start")
        self.save_btn.setEnabled(bool(self._segments))
        self.save_srt_btn.setEnabled(bool(self._segments))
        self.trans_progress.setVisible(False)
        self.trans_progress_label.setText("")

    @Slot(str)
    def _on_file_error(self, msg: str):
        QMessageBox.critical(self, "Transcription Error", msg)
        self._stop()

    # ── Live mode ───────────────────────────────────────────────────────
    def _start_live_mode(self):
        mode_id = self._mode_group.checkedId()
        is_mic = mode_id == 2
        
        device_idx = self.device_combo.currentData()
        if device_idx is None:
            QMessageBox.warning(self, "No device", "No device available.")
            self._stop()
            return

        self._ring = AudioRingBuffer(self.metrics)
        self._live_worker = WhisperWorker(self._transcriber, self._ring, self.metrics)
        self._live_worker.result.connect(self._on_live_result)
        self._live_worker.finished.connect(self._on_live_worker_done)
        self._live_committed = ""

        self.statusBar().showMessage("Listening...")
        try:
            if is_mic:
                self._capture.start_microphone(device_idx, self._on_audio_chunk)
            else:
                self._capture.start(device_idx, self._on_audio_chunk)
        except Exception as e:
            QMessageBox.critical(self, "Audio Error", f"Failed to start capture:\n{e}")
            self._stop()
            return

        self._live_worker.start()

    def _on_audio_chunk(self, chunk: np.ndarray):
        if self._ring:
            self._ring.push(chunk)

    # ── Performance metrics ─────────────────────────────────────────────
    def _update_status_metrics(self):
        """Called every 1s: refresh latency/queue/cpu/rate labels, push log lines to the console."""
        s = self.metrics.snapshot()
        if s["ring_pending_sec"] > 0:
            self.latency_label.setText(f"behind {s['ring_pending_sec']:.1f}s")
        else:
            self.latency_label.setText("behind —")
        if s["ring_pending_sec"] > 0:
            self.queue_label.setText(f"queue {s['ring_pending_sec']:.1f}s")
        else:
            self.queue_label.setText("queue —")
        if psutil:
            self.cpu_label.setText(f"cpu {psutil.cpu_percent():.0f}%")
        else:
            self.cpu_label.setText("cpu —")
        if s["capture_per_s"] > 0:
            self.rate_label.setText(f"rate {s['capture_per_s']:.0f} ch/s")
        else:
            self.rate_label.setText("rate —")
        if s["inference_events"] > 0 and s["inference_ms_avg"] > 0:
            rt = s["inference_ms_avg"] / 1000 / max(
                s["audio_processed_sec"] / max(s["inference_events"], 1), 1e-9
            )
            self.metrics.log(
                f"window {s['window_sec']:.0f}s | infer avg {s['inference_ms_avg']:.0f}ms "
                f"({rt:.2f}x RT) | capture avg {s['capture_ms_avg']:.0f}ms | "
                f"resample avg {s['resample_ms_avg']:.0f}ms | behind {s['ring_pending_sec']:.1f}s "
                f"| {s['capture_per_s']:.0f} ch/s | ring drops {s['ring_dropped_sec']:.1f}s"
            )
        self.console and self.console.drain_logs()

    def _on_live_worker_done(self):
        pass

    @Slot(object)
    def _on_live_result(self, result):
        start, end, committed, tentative = result
        self.metrics.log(f"[ui] live result: committed={committed!r} tentative={tentative!r}")
        if committed:
            self._live_committed += committed + " "
        self._update_live_display(tentative)

    def _update_live_display(self, tentative: str = ""):
        committed = self._live_committed.strip()
        self.transcript.clear()
        t = THEMES[self._theme]
        if committed:
            self.transcript.append(f'<span style="color:{t["text"]};">{committed}</span>')
        self.metrics.log(f"[ui] display updated: committed={committed!r} tentative={tentative!r}")

        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.transcript.setTextCursor(cursor)

    # ── Save ────────────────────────────────────────────────────────────
    def _plain_transcript_text(self) -> str:
        """Plain transcript text without timestamps."""
        if self._segments:
            return "\n".join(text for _, _, text in self._segments)
        return self.transcript.toPlainText()

    def _save_transcript(self):
        text = self._plain_transcript_text()
        if not text.strip():
            return
        # Default to same name/location as the source file
        if self._selected_file:
            base = os.path.splitext(self._selected_file)[0]
            default_path = base + ".txt"
        else:
            default_path = ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript", default_path,
            "Text files (*.txt);;All files (*.*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.statusBar().showMessage(f"Saved to {os.path.basename(path)}")

    def _save_srt(self):
        if not self._segments:
            return
        if self._selected_file:
            base = os.path.splitext(self._selected_file)[0]
            default_path = base + ".srt"
        else:
            default_path = ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SRT Subtitles", default_path,
            "SRT files (*.srt);;All files (*.*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                for i, (start, end, text) in enumerate(self._segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._srt_ts(start)} --> {self._srt_ts(end)}\n")
                    f.write(f"{text}\n\n")
            self.statusBar().showMessage(f"Saved to {os.path.basename(path)}")

    @staticmethod
    def _srt_ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ── Helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ── Debug console ──────────────────────────────────────────────────────
class DebugConsole(QWidget):
    """Live performance readout + rolling log of inference passes."""

    def __init__(self, metrics: Metrics):
        super().__init__()
        self.metrics = metrics
        self.setWindowTitle("Debug Console")
        self.resize(760, 480)

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumHeight(240)
        layout.addWidget(self.log, 1)

        self._timer = QTimer()
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)

    def refresh(self):
        s = self.metrics.snapshot()
        rows = [
            ("Latency (ring pending)", f"{s['ring_pending_sec']:.2f} s"),
            ("Max ring pending", f"{s['max_ring_pending_sec']:.2f} s"),
            ("Ring dropped audio", f"{s['ring_dropped_sec']:.1f} s"),
            ("VAD silence drained", f"{s['silence_drained']:.1f} s"),
            ("VAD skips", str(s["silence_skips"])),
            ("Inference / pass", f"{s['inference_ms_avg']:.0f} ms"),
            ("Inference events", str(s["inference_events"])),
            ("RT factor", f"{s['inference_ms_avg'] / 1000 / max(s['audio_processed_sec'] / max(s['inference_events'], 1), 1e-9):.2f}x"),
            ("Capture / chunk", f"{s['capture_ms_avg']:.1f} ms"),
            ("Resample / chunk", f"{s['resample_ms_avg']:.1f} ms"),
            ("Chunks processed / s", f"{s['capture_per_s']:.0f}"),
            ("Audio received", f"{s['audio_received_sec']:.1f} s"),
            ("Audio processed", f"{s['audio_processed_sec']:.1f} s"),
            ("CPU (process)", f"{psutil.Process().cpu_percent():.0f}%" if psutil else "n/a"),
            ("Skipped drops", str(s["skipped_drops"])),
        ]
        self.table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(k))
            self.table.setItem(r, 1, QTableWidgetItem(v))
        self.drain_logs()

    def drain_logs(self):
        for ts, msg in self.metrics.drain_log():
            self.log.append(f"[{self.metrics.format_ts(ts)}] {msg}")
            if self.log.document().blockCount() > 2000:
                self.log.clear()

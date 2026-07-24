import os
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from transcriber import Transcriber

# ── Colors ──────────────────────────────────────────────────────────────
BG       = "#0f0f1a"
BG2      = "#16162a"
CARD     = "#1c1c35"
CARD_HL  = "#22223d"
BORDER   = "#2a2a4a"
ACCENT   = "#7c6fff"
ACCENT2  = "#a78bfa"
GREEN    = "#4ade80"
YELLOW   = "#facc15"
RED      = "#f87171"
TXT      = "#e2e8f0"
TXT2     = "#94a3b8"
TXT3     = "#64748b"


class CircularProgress(tk.Canvas):
    """Animated circular progress indicator."""

    def __init__(self, parent, size=120, width=8, **kw):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bg=BG2, **kw)
        self.size = size
        self.width = width
        self.center = size // 2
        self.radius = (size - width) // 2
        self._angle = 0
        self._pct = 0
        self._animating = False

    def set_progress(self, pct):
        self._pct = pct
        self._draw()

    def start_spin(self):
        self._animating = True
        self._spin()

    def stop_spin(self):
        self._animating = False

    def _spin(self):
        if not self._animating:
            return
        self._angle = (self._angle + 4) % 360
        self._draw_arc()
        self.after(16, self._spin)

    def _draw(self):
        self.delete("all")
        # Background ring
        self.create_arc(
            self.width, self.width,
            self.size - self.width, self.size - self.width,
            start=90, extent=359.9,
            outline=BORDER, width=self.width, style="arc",
        )
        # Progress arc
        extent = self._pct * 3.6
        if extent > 0:
            self.create_arc(
                self.width, self.width,
                self.size - self.width, self.size - self.width,
                start=90, extent=-extent,
                outline=ACCENT, width=self.width, style="arc",
            )
        # Percentage text
        self.create_text(
            self.center, self.center - 8,
            text=f"{int(self._pct)}%",
            fill=TXT, font=("Segoe UI", 18, "bold"),
        )
        self.create_text(
            self.center, self.center + 16,
            text="transcribing",
            fill=TXT3, font=("Segoe UI", 9),
        )

    def _draw_arc(self):
        self.delete("all")
        # Spinning arc
        self.create_arc(
            self.width, self.width,
            self.size - self.width, self.size - self.width,
            start=self._angle, extent=120,
            outline=ACCENT, width=self.width, style="arc",
        )
        self.create_text(
            self.center, self.center - 8,
            text="Loading...",
            fill=TXT, font=("Segoe UI", 14, "bold"),
        )
        self.create_text(
            self.center, self.center + 14,
            text="preparing model",
            fill=TXT3, font=("Segoe UI", 9),
        )


class RoundedButton(tk.Canvas):
    """A custom rounded button with hover effects."""

    def __init__(self, parent, text, command=None, width=180, height=44,
                 bg=ACCENT, fg="white", hover_bg=None, font_size=11, **kw):
        super().__init__(parent, width=width, height=height,
                         highlightthickness=0, bg=BG2, **kw)
        self.command = command
        self._bg = bg
        self._fg = fg
        self._hover_bg = hover_bg or self._lighten(bg)
        self._width = width
        self._height = height
        self._font_size = font_size
        self._draw(text, bg, fg)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self, text, bg, fg):
        self.delete("all")
        r = 10
        w, h = self._width, self._height
        self.create_arc(0, 0, 2 * r, 2 * r, start=90, extent=90, fill=bg, outline="")
        self.create_arc(w - 2 * r, 0, w, 2 * r, start=0, extent=90, fill=bg, outline="")
        self.create_arc(0, h - 2 * r, 2 * r, h, start=180, extent=90, fill=bg, outline="")
        self.create_arc(w - 2 * r, h - 2 * r, w, h, start=270, extent=90, fill=bg, outline="")
        self.create_rectangle(r, 0, w - r, h, fill=bg, outline="")
        self.create_rectangle(0, r, w, h - r, fill=bg, outline="")
        self.create_text(w // 2, h // 2, text=text, fill=fg,
                         font=("Segoe UI", self._font_size, "bold"))

    def _lighten(self, color):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        r, g, b = min(255, r + 30), min(255, g + 30), min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_enter(self, e):
        self._draw(self._get_text(), self._hover_bg, self._fg)

    def _on_leave(self, e):
        self._draw(self._get_text(), self._bg, self._fg)

    def _on_press(self, e):
        self._draw(self._get_text(), self._darken(self._bg), self._fg)

    def _on_release(self, e):
        self._draw(self._get_text(), self._bg, self._fg)
        if self.command:
            self.command()

    def _darken(self, color):
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        r, g, b = max(0, r - 20), max(0, g - 20), max(0, b - 20)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _get_text(self):
        for item in self.find_all():
            if self.type(item) == "text":
                return self.itemcget(item, "text")
        return ""


class TranscriberApp:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("Transcriber")
        self.root.geometry("860x600")
        self.root.minsize(650, 450)
        self.root.configure(bg=BG)

        self.transcriber = None
        self.current_source = None
        self.segments = None

        self._build_ui()

    def _build_ui(self):
        # ── Welcome / Drop Frame ────────────────────────────────────────
        self.drop_frame = tk.Frame(self.root, bg=BG)
        self.drop_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Icon
        icon_frame = tk.Frame(self.drop_frame, bg=BG)
        icon_frame.pack(pady=(0, 20))
        tk.Label(icon_frame, text="\U0001f399", font=("Segoe UI", 52),
                 bg=BG, fg=ACCENT).pack()

        tk.Label(self.drop_frame, text="Transcriber", font=("Segoe UI", 28, "bold"),
                 bg=BG, fg=TXT).pack(pady=(0, 6))
        tk.Label(self.drop_frame, text="Drop a file or browse to start",
                 font=("Segoe UI", 12), bg=BG, fg=TXT2).pack(pady=(0, 4))
        tk.Label(self.drop_frame, text="Supports MP3, WAV, MP4, MKV, and more",
                 font=("Segoe UI", 10), bg=BG, fg=TXT3).pack(pady=(0, 28))

        self.browse_btn = RoundedButton(
            self.drop_frame, text="Browse Files", command=self._pick_file,
            width=200, height=48, bg=ACCENT, font_size=12,
        )
        self.browse_btn.pack(pady=(0, 20))

        # Model status
        model_frame = tk.Frame(self.drop_frame, bg=CARD, padx=14, pady=8)
        model_frame.pack()
        self.model_dot = tk.Canvas(model_frame, width=10, height=10,
                                    highlightthickness=0, bg=CARD)
        self.model_dot.pack(side="left", padx=(0, 8))
        self.model_dot.create_oval(2, 2, 8, 8, fill=TXT3, outline="")
        self.model_label = tk.Label(model_frame, text="Model: Not loaded",
                                     font=("Segoe UI", 9), bg=CARD, fg=TXT3)
        self.model_label.pack(side="left")

        # ── Drag-and-Drop ──────────────────────────────────────────────
        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind("<<DropEnter>>", self._on_drop_enter)
        self.drop_frame.dnd_bind("<<DropLeave>>", self._on_drop_leave)
        self.drop_frame.dnd_bind("<<Drop>>", self._on_drop)

        # Also allow dropping on the whole window as fallback
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

        # ── Transcription Frame ─────────────────────────────────────────
        self.trans_frame = tk.Frame(self.root, bg=BG)

        # Top bar: file info + new file button
        top_bar = tk.Frame(self.trans_frame, bg=BG)
        top_bar.pack(fill="x", padx=20, pady=(16, 8))

        file_frame = tk.Frame(top_bar, bg=CARD, padx=12, pady=8)
        file_frame.pack(side="left", fill="x", expand=True)
        self.file_icon = tk.Label(file_frame, text="\U0001f3b5", font=("Segoe UI", 14),
                                   bg=CARD, fg=ACCENT)
        self.file_icon.pack(side="left", padx=(0, 8))
        self.file_label = tk.Label(file_frame, text="", font=("Segoe UI", 11, "bold"),
                                    bg=CARD, fg=TXT, anchor="w")
        self.file_label.pack(side="left", fill="x", expand=True)

        self.new_file_btn = RoundedButton(
            top_bar, text="New File", command=self._reset_to_drop,
            width=100, height=36, bg=CARD_HL, fg=TXT2, font_size=9,
            hover_bg=BORDER,
        )
        self.new_file_btn.pack(side="right", padx=(10, 0))

        # Progress area
        self.progress_frame = tk.Frame(self.trans_frame, bg=BG2)
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.progress_bar = tk.Canvas(self.progress_frame, height=6,
                                       highlightthickness=0, bg=BG2)
        self.progress_bar.pack(fill="x", padx=0, pady=(12, 0))
        self.progress_bar.bind("<Configure>", self._draw_progress_bg)

        self.progress_label = tk.Label(self.progress_frame, text="Transcribing...",
                                        font=("Segoe UI", 10), bg=BG2, fg=TXT2)
        self.progress_label.pack(padx=16, pady=(6, 12), anchor="w")

        # Circular progress (shown during model loading)
        self.circular = CircularProgress(self.trans_frame, size=140, width=10)
        self.circular_frame = tk.Frame(self.trans_frame, bg=BG)
        self._circular_center = None

        # Text preview
        text_outer = tk.Frame(self.trans_frame, bg=CARD, padx=1, pady=1)
        text_outer.pack(expand=True, fill="both", padx=20, pady=(0, 10))

        self.preview = tk.Text(text_outer, bg=BG2, fg=TXT,
                                font=("Cascadia Code", 10), wrap="word",
                                insertbackground=TXT, selectbackground="#3b3b6d",
                                relief="flat", padx=16, pady=12,
                                spacing1=2, spacing2=1, spacing3=2)
        scrollbar = tk.Scrollbar(text_outer, command=self.preview.yview,
                                  bg=CARD, troughcolor=BG2, width=8)
        self.preview.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self.preview.pack(expand=True, fill="both", padx=4, pady=4)
        self.preview.configure(state="disabled")

        # Export bar
        export_bar = tk.Frame(self.trans_frame, bg=BG)
        export_bar.pack(fill="x", padx=20, pady=(0, 16))

        self.export_txt_btn = RoundedButton(
            export_bar, text="  Export TXT  ", command=self._export_txt,
            width=130, height=40, bg="#166534", fg=GREEN, font_size=10,
            hover_bg="#15803d",
        )
        self.export_txt_btn.pack(side="left", padx=(0, 8))
        self.export_txt_btn.configure(state="disabled")

        self.export_srt_btn = RoundedButton(
            export_bar, text="  Export SRT  ", command=self._export_srt,
            width=130, height=40, bg="#1e3a5f", fg="#60a5fa", font_size=10,
            hover_bg="#2563eb",
        )
        self.export_srt_btn.pack(side="left")
        self.export_srt_btn.configure(state="disabled")

        self.status_msg = tk.Label(export_bar, text="", font=("Segoe UI", 9, "bold"),
                                    bg=BG, fg=GREEN)
        self.status_msg.pack(side="right", padx=(10, 0))

    # ── Progress bar ────────────────────────────────────────────────────
    def _draw_progress_bg(self, event=None):
        w = self.progress_bar.winfo_width()
        h = 6
        self.progress_bar.delete("all")
        self.progress_bar.create_rectangle(0, 0, w, h, fill=BORDER, outline="")

    def _update_progress(self, pct):
        w = self.progress_bar.winfo_width()
        h = 6
        self.progress_bar.delete("all")
        # Background
        self.progress_bar.create_rectangle(0, 0, w, h, fill=BORDER, outline="")
        # Fill with gradient effect
        fill_w = max(1, int(w * pct))
        for i in range(fill_w):
            ratio = i / max(1, fill_w)
            r = int(124 + (167 - 124) * ratio)
            g = int(111 + (139 - 111) * ratio)
            b = int(255 + (250 - 255) * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.progress_bar.create_line(i, 0, i, h, fill=color, width=1)
        self.progress_label.configure(text=f"Transcribing... {int(pct * 100)}%")

    # ── Drag-and-Drop ──────────────────────────────────────────────────
    def _on_drop_enter(self, e):
        self.drop_frame.configure(bg=CARD_HL)
        for child in self.drop_frame.winfo_children():
            try:
                child.configure(bg=CARD_HL)
            except tk.TclError:
                pass

    def _on_drop_leave(self, e):
        self.drop_frame.configure(bg=BG)
        for child in self.drop_frame.winfo_children():
            try:
                child.configure(bg=BG)
            except tk.TclError:
                pass

    def _on_drop(self, e):
        self.drop_frame.configure(bg=BG)
        for child in self.drop_frame.winfo_children():
            try:
                child.configure(bg=BG)
            except tk.TclError:
                pass

        files = self.root.splitlist(e.data)
        if files:
            path = files[0]
            ext = os.path.splitext(path)[1].lower()
            supported = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
                         ".aac", ".mp4", ".mkv", ".avi", ".mov", ".webm"}
            if ext in supported:
                self._start_transcription(path)
            else:
                messagebox.showwarning("Unsupported format",
                                       f"Cannot transcribe {ext} files.\n"
                                       f"Supported: MP3, WAV, M4A, FLAC, OGG, MP4, MKV, etc.")

    # ── Actions ─────────────────────────────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select audio or video file",
            filetypes=[
                ("Audio/Video", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac"),
                ("Video", "*.mp4 *.mkv *.avi *.mov *.webm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._start_transcription(path)

    def _start_transcription(self, file_path):
        self.current_source = file_path
        name = os.path.basename(file_path)
        self.file_label.configure(text=name)
        self.file_icon.configure(text="\U0001f3b5" if any(
            name.lower().endswith(e) for e in (".mp3", ".wav", ".m4a", ".flac", ".ogg")
        ) else "\U0001f3ac")

        # Switch views
        self.drop_frame.place_forget()
        self.trans_frame.pack(expand=True, fill="both")

        # Reset
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.configure(state="disabled")
        self.export_txt_btn.configure(state="disabled")
        self.export_srt_btn.configure(state="disabled")
        self.progress_bar.delete("all")
        self.progress_label.configure(text="Transcribing...")
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))

        threading.Thread(target=self._transcribe_thread, args=(file_path,), daemon=True).start()

    def _transcribe_thread(self, file_path):
        try:
            self._ensure_model()

            def progress_cb(pct):
                self.root.after(0, self._update_progress, pct)

            self.segments, full_text = self.transcriber.transcribe(
                file_path, progress_callback=progress_cb
            )
            self.root.after(0, self._transcription_done, full_text)
        except Exception as e:
            self.root.after(0, self._transcription_error, str(e))

    def _ensure_model(self):
        if self.transcriber is None:
            self.root.after(0, self._set_model_status, "loading",
                            "Loading model...")
            self.root.after(0, self._show_circular)
            self.transcriber = Transcriber("tiny.en")
            self.root.after(0, self._hide_circular)
            self.root.after(0, self._set_model_status, "loaded",
                            "Model: tiny.en ready")

    def _show_circular(self):
        self.progress_frame.pack_forget()
        self.circular_frame = tk.Frame(self.trans_frame, bg=BG)
        self.circular_frame.pack(expand=True)
        self.circular.place(in_=self.circular_frame, relx=0.5, rely=0.5, anchor="center")
        self.circular.start_spin()

    def _hide_circular(self):
        self.circular.stop_spin()
        self.circular.place_forget()
        self.circular_frame.destroy()
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))

    def _set_model_status(self, state, text):
        color_map = {"loading": YELLOW, "loaded": GREEN, "error": RED}
        fill_map = {"loading": YELLOW, "loaded": GREEN, "error": RED}
        self.model_label.configure(text=text, fg=color_map.get(state, TXT3))
        self.model_dot.delete("all")
        self.model_dot.create_oval(2, 2, 8, 8, fill=fill_map.get(state, TXT3), outline="")

    def _transcription_done(self, text):
        self.progress_frame.pack_forget()

        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

        self.export_txt_btn.configure(state="normal")
        self.export_srt_btn.configure(state="normal")

    def _transcription_error(self, msg):
        self.progress_frame.pack_forget()
        messagebox.showerror("Error", msg)
        self._reset_to_drop()

    def _export_txt(self):
        if not self.segments or not self.current_source:
            return
        base = os.path.splitext(self.current_source)[0]
        out_path = base + ".txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.transcriber.generate_txt(self.segments))
        self._show_status(f"Saved: {os.path.basename(out_path)}", GREEN)

    def _export_srt(self):
        if not self.segments or not self.current_source:
            return
        base = os.path.splitext(self.current_source)[0]
        out_path = base + ".srt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.transcriber.generate_srt(self.segments))
        self._show_status(f"Saved: {os.path.basename(out_path)}", "#60a5fa")

    def _show_status(self, msg, color):
        self.status_msg.configure(text=msg, fg=color)
        self.root.after(3000, lambda: self.status_msg.configure(text=""))

    def _reset_to_drop(self):
        self.trans_frame.pack_forget()
        self.drop_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.configure(state="disabled")
        self.export_txt_btn.configure(state="disabled")
        self.export_srt_btn.configure(state="disabled")
        self.progress_bar.delete("all")
        self.progress_var = 0

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    TranscriberApp().run()

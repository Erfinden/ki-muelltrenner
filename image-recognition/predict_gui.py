#!/usr/bin/env python3
"""
predict_gui.py
==============
KI-Mülltrenner – GUI Prediction Tool

Shows a live webcam preview.  Clicking "Prüfen" captures a frame, saves it to
captures/ and runs it through the trained FastAI model.  The prediction
probabilities for every label are displayed in coloured progress bars.

Optionally connects to the Arduino over USB serial and sends an OPEN command
to the correct lid based on the top prediction.

Usage:
    python predict_gui.py

Dependencies (same venv as the rest of the project):
    fastai, opencv-python, Pillow, torch, tkinter (stdlib), pyserial
"""

import os
import re
import sys
import cv2
import time
import shutil
import threading
import contextlib
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_camera_errors():
    """Silence C-level stdout/stderr (OpenCV out-of-bound messages) during camera probing."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved = [os.dup(1), os.dup(2)]
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(devnull_fd)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH         = Path(__file__).parent / "models" / "trash_classifier.pkl"
CAPTURES_DIR       = Path(__file__).parent / "captures"
COLLECTED_DATA_DIR = Path(__file__).parent / "collected-data"

# Path to the shared TrashBinController module (one level up from image-recognition/)
_CONTROLLER_DIR = Path(__file__).parent.parent / "arduino-sketch"
if str(_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_DIR))

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ── Label → Lid mapping ───────────────────────────────────────────────────────
# Lid 1 = Papier/paper,  Lid 2 = everything else
LABEL_TO_LID: dict[str, int] = {
    "papier":   1,
    "paper":    1,
    # all other labels default to lid 2 (see _label_to_lid())
}

def _label_to_lid(label: str) -> int:
    return LABEL_TO_LID.get(label.lower(), 2)


# Accent colours per label slot (up to 10 labels; cycles if more)
SLOT_COLORS = [
    "#4ade80",  # green
    "#facc15",  # yellow
    "#60a5fa",  # blue
    "#f87171",  # red
    "#a78bfa",  # violet
    "#34d399",  # emerald
    "#fb923c",  # orange
    "#38bdf8",  # sky
    "#e879f9",  # fuchsia
    "#2dd4bf",  # teal
]

BG_DARK      = "#1a1a2e"
BG_PANEL     = "#16213e"
BG_CARD      = "#0f3460"
FG_TEXT      = "#e2e8f0"
FG_MUTED     = "#94a3b8"
ACCENT       = "#e94560"
ACCENT_HOVER = "#c73652"
BTN_TEXT     = "#000000"

COLOR_CONNECTED    = "#4ade80"   # green dot
COLOR_DISCONNECTED = "#f87171"   # red dot

# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Load the FastAI learner.  Returns (learner, vocab) or raises."""
    from fastai.vision.all import load_learner
    learn = load_learner(MODEL_PATH)
    vocab = list(learn.dls.vocab)
    return learn, vocab


def _predict(learner, image_path: Path):
    """Run inference on *image_path*.  Returns list of (label, pct) tuples."""
    from fastai.vision.all import PILImage
    img = PILImage.create(image_path)
    _, _, probs = learner.predict(img)
    vocab = list(learner.dls.vocab)
    results = [(vocab[i], float(probs[i]) * 100) for i in range(len(vocab))]
    results.sort(key=lambda x: -x[1])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────

class PredictGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KI-Mülltrenner – Prüfen")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # ── Model / camera state ──────────────────────────────────────────────
        self.learner    = None
        self.vocab: list[str] = []
        self.current_frame_bgr = None
        self.photo_image       = None
        self.cap               = None
        self.current_camera_index = 0
        self.available_cameras    = []

        # ── Last prediction (for feedback buttons) ────────────────────────────
        self._last_capture_path: Path | None = None
        self._last_top_label:    str  | None = None
        self._last_results:      list        = []

        # ── Arduino state ─────────────────────────────────────────────────────
        self._arduino = None          # TrashBinController instance (or None)
        self._arduino_lock = threading.Lock()
        self._led_on = False          # LED strip state (off by default)

        # ── Label widgets (populated after model load) ────────────────────────
        self._bar_vars:   dict[str, tk.DoubleVar]    = {}
        self._pct_vars:   dict[str, tk.StringVar]    = {}
        self._bar_frames: dict[str, ttk.Progressbar] = {}

        # Build UI skeleton (video + right panel)
        self._build_ui()
        self._bring_to_front()

        # Load model then open camera
        self.root.after(100, self._startup)

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Main two-column grid
        outer = tk.Frame(self.root, bg=BG_DARK)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        # ── Left: video + controls ────────────────────────────────────────────
        left = tk.Frame(outer, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        video_card = tk.Frame(left, bg=BG_PANEL, bd=0, relief=tk.FLAT)
        video_card.grid(row=0, column=0, sticky="nsew")

        self.video_label = tk.Label(video_card, bg=BG_PANEL, cursor="none")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Camera selector row (always visible)
        cam_row = tk.Frame(left, bg=BG_DARK)
        cam_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        cam_row.columnconfigure(1, weight=1)

        tk.Label(
            cam_row, text="Kamera:", bg=BG_DARK, fg=FG_MUTED,
            font=("Helvetica", 10), anchor="w"
        ).grid(row=0, column=0, sticky="w")

        self.camera_var = tk.StringVar()
        self._cam_combo = ttk.Combobox(
            cam_row, textvariable=self.camera_var,
            state="readonly", width=14
        )
        self._cam_combo.grid(row=0, column=1, sticky="ew", padx=(6, 4))
        self._cam_combo.bind("<<ComboboxSelected>>", self._switch_camera)

        tk.Button(
            cam_row, text="↻", font=("Helvetica", 11, "bold"),
            bg=BG_PANEL, fg=FG_MUTED, relief=tk.FLAT, bd=0,
            cursor="hand2", padx=4, pady=2,
            command=self._refresh_cameras
        ).grid(row=0, column=2, sticky="e")

        # "Prüfen" button
        self._build_pruefen_button(left)

        # Status bar (row 4 because feedback row takes row 3)
        self.status_var = tk.StringVar(value="Modell wird geladen …")
        status_bar = tk.Label(
            left, textvariable=self.status_var,
            bg=BG_DARK, fg=FG_MUTED, font=("Helvetica", 10),
            anchor="w"
        )
        status_bar.grid(row=4, column=0, sticky="ew", pady=(4, 0))

        # ── Right: labels panel + Arduino panel ───────────────────────────────
        right = tk.Frame(outer, bg=BG_PANEL, padx=14, pady=14)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        hdr = tk.Label(
            right, text="Ergebnisse", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 14, "bold"), anchor="w"
        )
        hdr.pack(fill=tk.X, pady=(0, 10))

        self.results_frame = tk.Frame(right, bg=BG_PANEL)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        loading_lbl = tk.Label(
            self.results_frame,
            text="Bitte warten…\nModell wird geladen.",
            bg=BG_PANEL, fg=FG_MUTED,
            font=("Helvetica", 11),
            justify="center"
        )
        loading_lbl.pack(pady=40)
        self._loading_lbl = loading_lbl

        # ── Arduino connection panel ───────────────────────────────────────────
        self._build_arduino_panel(right)

    def _build_pruefen_button(self, parent):
        """Create the big 'Prüfen' button and the Richtig/Falsch feedback row."""
        self.pruefen_btn = tk.Button(
            parent,
            text="📸  Prüfen",
            font=("Helvetica", 18, "bold"),
            bg=ACCENT,
            fg=BTN_TEXT,
            activebackground=ACCENT_HOVER,
            activeforeground=BTN_TEXT,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=20,
            pady=14,
            state=tk.DISABLED,
            command=self._on_pruefen,
        )
        self.pruefen_btn.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.pruefen_btn.bind("<Enter>", lambda e: self.pruefen_btn.config(bg=ACCENT_HOVER))
        self.pruefen_btn.bind("<Leave>", lambda e: self.pruefen_btn.config(bg=ACCENT))

        # ── Richtig / Falsch feedback row ─────────────────────────────────────
        fb_row = tk.Frame(parent, bg=BG_DARK)
        fb_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        fb_row.columnconfigure(0, weight=1)
        fb_row.columnconfigure(1, weight=1)

        COLOR_RICHTIG       = "#166534"
        COLOR_RICHTIG_HOVER = "#14532d"
        COLOR_FALSCH        = "#7f1d1d"
        COLOR_FALSCH_HOVER  = "#6b1919"

        self.richtig_btn = tk.Button(
            fb_row,
            text="✅  Richtig",
            font=("Helvetica", 13, "bold"),
            bg=COLOR_RICHTIG, fg=BTN_TEXT,
            activebackground=COLOR_RICHTIG_HOVER, activeforeground=BTN_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=10, pady=8,
            state=tk.DISABLED,
            command=self._on_richtig,
        )
        self.richtig_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.richtig_btn.bind("<Enter>", lambda e: self.richtig_btn.config(bg=COLOR_RICHTIG_HOVER) if self.richtig_btn["state"] == tk.NORMAL else None)
        self.richtig_btn.bind("<Leave>", lambda e: self.richtig_btn.config(bg=COLOR_RICHTIG))

        self.falsch_btn = tk.Button(
            fb_row,
            text="❌  Falsch",
            font=("Helvetica", 13, "bold"),
            bg=COLOR_FALSCH, fg=BTN_TEXT,
            activebackground=COLOR_FALSCH_HOVER, activeforeground=BTN_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=10, pady=8,
            state=tk.DISABLED,
            command=self._on_falsch,
        )
        self.falsch_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        self.falsch_btn.bind("<Enter>", lambda e: self.falsch_btn.config(bg=COLOR_FALSCH_HOVER) if self.falsch_btn["state"] == tk.NORMAL else None)
        self.falsch_btn.bind("<Leave>", lambda e: self.falsch_btn.config(bg=COLOR_FALSCH))

    def _build_arduino_panel(self, parent):
        """Build the Arduino serial connection card at the bottom of the right panel."""
        sep = tk.Frame(parent, bg=BG_CARD, height=1)
        sep.pack(fill=tk.X, pady=(16, 0))

        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=10)
        card.pack(fill=tk.X, pady=(6, 0))
        card.columnconfigure(1, weight=1)

        # Header row with status dot
        hdr_row = tk.Frame(card, bg=BG_CARD)
        hdr_row.pack(fill=tk.X)

        tk.Label(
            hdr_row, text="🔌  Arduino", bg=BG_CARD, fg=FG_TEXT,
            font=("Helvetica", 12, "bold"), anchor="w"
        ).pack(side=tk.LEFT)

        self._ard_status_dot = tk.Label(
            hdr_row, text="●", bg=BG_CARD, fg=COLOR_DISCONNECTED,
            font=("Helvetica", 14)
        )
        self._ard_status_dot.pack(side=tk.LEFT, padx=(6, 0))

        self._ard_status_lbl = tk.Label(
            hdr_row, text="Nicht verbunden", bg=BG_CARD, fg=FG_MUTED,
            font=("Helvetica", 10)
        )
        self._ard_status_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # Port row
        port_row = tk.Frame(card, bg=BG_CARD)
        port_row.pack(fill=tk.X, pady=(8, 0))

        tk.Label(
            port_row, text="Port:", bg=BG_CARD, fg=FG_MUTED,
            font=("Helvetica", 10), width=5, anchor="w"
        ).pack(side=tk.LEFT)

        self._port_var = tk.StringVar(value="")
        self._port_combo = ttk.Combobox(
            port_row, textvariable=self._port_var,
            width=20, font=("Helvetica", 10)
        )
        self._port_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

        refresh_btn = tk.Button(
            port_row, text="↻", font=("Helvetica", 11, "bold"),
            bg=BG_PANEL, fg=FG_MUTED, relief=tk.FLAT, bd=0,
            cursor="hand2", padx=4, pady=2,
            command=self._refresh_ports
        )
        refresh_btn.pack(side=tk.LEFT)

        # Connect / Disconnect button
        self._connect_btn = tk.Button(
            card,
            text="Verbinden",
            font=("Helvetica", 11, "bold"),
            bg="#1e3a5f", fg=BTN_TEXT,
            activebackground="#2a4f80",
            activeforeground=BTN_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=10, pady=6,
            command=self._toggle_arduino_connection
        )
        self._connect_btn.pack(fill=tk.X, pady=(8, 0))

        # Checkbox – send to Arduino
        self._send_to_arduino_var = tk.BooleanVar(value=False)
        self._send_chk = tk.Checkbutton(
            card,
            text="Vorhersage an Arduino senden",
            variable=self._send_to_arduino_var,
            bg=BG_CARD, fg=FG_TEXT, selectcolor=BG_PANEL,
            activebackground=BG_CARD, activeforeground=FG_TEXT,
            font=("Helvetica", 11),
            anchor="w", cursor="hand2",
        )
        self._send_chk.pack(fill=tk.X, pady=(8, 0))

        # ── Manual test buttons ───────────────────────────────────────────────
        tk.Frame(card, bg=BG_PANEL, height=1).pack(fill=tk.X, pady=(10, 0))
        tk.Label(
            card, text="Manueller Test", bg=BG_CARD, fg=FG_MUTED,
            font=("Helvetica", 9), anchor="w"
        ).pack(fill=tk.X, pady=(6, 2))

        lid_row = tk.Frame(card, bg=BG_CARD)
        lid_row.pack(fill=tk.X)
        lid_row.columnconfigure(0, weight=1)
        lid_row.columnconfigure(1, weight=1)

        self._lid1_btn = tk.Button(
            lid_row,
            text="🗑  Deckel 1",
            font=("Helvetica", 11, "bold"),
            bg="#1a3a4a", fg=BTN_TEXT,
            activebackground="#1f4f66", activeforeground=BTN_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=6, pady=6,
            state=tk.DISABLED,
            command=lambda: self._manual_open_lid(1),
        )
        self._lid1_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=(0, 0))

        self._lid2_btn = tk.Button(
            lid_row,
            text="🗑  Deckel 2",
            font=("Helvetica", 11, "bold"),
            bg="#1a3a4a", fg=BTN_TEXT,
            activebackground="#1f4f66", activeforeground=BTN_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=6, pady=6,
            state=tk.DISABLED,
            command=lambda: self._manual_open_lid(2),
        )
        self._lid2_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        # ── LED light toggle ─────────────────────────────────────────────────
        tk.Frame(card, bg=BG_PANEL, height=1).pack(fill=tk.X, pady=(8, 0))
        self._light_btn = tk.Button(
            card,
            text="💡  Licht: AUS",
            font=("Helvetica", 11, "bold"),
            bg="#2a2a3a", fg=FG_MUTED,
            activebackground="#3a3a4a", activeforeground=FG_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=6, pady=6,
            state=tk.DISABLED,
            command=self._toggle_led,
        )
        self._light_btn.pack(fill=tk.X, pady=(6, 0))

        # Populate port list
        self._refresh_ports()

    # ──────────────────────────────────────────────────────────────────────────
    # Startup sequence
    # ──────────────────────────────────────────────────────────────────────────

    def _startup(self):
        """Load model and open camera (called once after the event loop starts)."""
        # 1. Load model
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Modell nicht gefunden",
                f"Kein Modell unter:\n{MODEL_PATH}\n\n"
                "Bitte zuerst train.py ausführen."
            )
            self.root.destroy()
            return

        try:
            self.status_var.set("Modell wird geladen …")
            self.root.update_idletasks()
            self.learner, self.vocab = _load_model()
        except Exception as exc:
            messagebox.showerror("Ladefehler", f"Modell konnte nicht geladen werden:\n{exc}")
            self.root.destroy()
            return

        # 2. Build label rows now that we know the vocab
        self._build_label_rows()

        # 3. Detect cameras in background so the UI stays responsive
        self.status_var.set("Suche Kameras …")
        self.root.update_idletasks()

        def _camera_worker():
            cameras = self._detect_cameras()
            self.root.after(0, lambda: self._on_cameras_ready(cameras))

        threading.Thread(target=_camera_worker, daemon=True).start()

    def _on_cameras_ready(self, cameras):
        """Called on the main thread after background camera detection finishes."""
        self.available_cameras = cameras
        if not cameras:
            messagebox.showerror("Kamera-Fehler", "Keine Kamera gefunden.")
            self.root.destroy()
            return

        # Populate selector & open first camera
        values = [f"Gerät {i}" for i in cameras]
        self._cam_combo.config(values=values)
        self.camera_var.set(values[0])
        self._open_camera(cameras[0])

        # Enable button & start preview loop
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        self.pruefen_btn.config(state=tk.NORMAL)
        self.status_var.set('Bereit – klicke "Prüfen" um ein Bild aufzunehmen.')
        self._update_frame()

    def _detect_cameras(self):
        """Probe indices 0-4. Each probe runs in its own thread with a 2-second
        timeout so a hanging VideoCapture call (common on some Windows setups)
        never blocks the application. Stops after 2 consecutive failures."""
        _backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cameras = []
        consecutive_fails = 0

        def _probe(idx):
            try:
                with _suppress_camera_errors():
                    cap = cv2.VideoCapture(idx, _backend)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        return bool(ret)
            except Exception:
                pass
            return False

        for i in range(5):
            found = [False]
            t = threading.Thread(
                target=lambda i=i: found.__setitem__(0, _probe(i)),
                daemon=True,
            )
            t.start()
            t.join(timeout=2.0)   # abandon the probe if it hangs
            if found[0]:
                cameras.append(i)
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                if consecutive_fails >= 2:
                    break

        return cameras if cameras else [0]

    def _open_camera(self, index: int):
        _backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, _backend)
        if not cap.isOpened():
            messagebox.showerror("Kamera-Fehler", f"Kamera {index} konnte nicht geöffnet werden.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap = cap
        self.current_camera_index = index

    def _refresh_cameras(self):
        """Re-detect cameras in the background and update the combobox."""
        self._cam_combo.config(state=tk.DISABLED)

        def _worker():
            cameras = self._detect_cameras()
            self.root.after(0, lambda: self._on_cameras_ready(cameras))

        threading.Thread(target=_worker, daemon=True).start()

    def _switch_camera(self, event=None):
        selected = self.camera_var.get()
        try:
            new_idx = int(selected.split()[-1])
        except (ValueError, IndexError):
            return
        if new_idx == self.current_camera_index:
            return

        # Disable combobox while switching so the user can't trigger it again
        self._cam_combo.config(state=tk.DISABLED)
        self.status_var.set(f"Wechsle zu Kamera {new_idx} …")

        def _worker():
            _backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            with _suppress_camera_errors():
                new_cap = cv2.VideoCapture(new_idx, _backend)
            if not new_cap.isOpened():
                self.root.after(0, lambda: self._on_camera_switch_failed(new_idx, "konnte nicht geöffnet werden"))
                return
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            ret, _ = new_cap.read()
            if not ret:
                new_cap.release()
                self.root.after(0, lambda: self._on_camera_switch_failed(new_idx, "liefert kein Bild"))
                return
            self.root.after(0, lambda: self._on_camera_switch_ready(new_idx, new_cap))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_camera_switch_ready(self, new_idx, new_cap):
        """Called on the main thread when the new camera opened successfully."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = new_cap
        self.current_camera_index = new_idx
        self._cam_combo.config(state="readonly")
        self.status_var.set('Bereit – klicke "Prüfen" um ein Bild aufzunehmen.')

    def _on_camera_switch_failed(self, new_idx, reason):
        """Called on the main thread when the new camera could not be opened."""
        messagebox.showerror("Kamera-Fehler", f"Kamera {new_idx} nicht verfügbar: {reason}.")
        self.camera_var.set(f"Gerät {self.current_camera_index}")
        self._cam_combo.config(state="readonly")
        self.status_var.set('Bereit – klicke "Prüfen" um ein Bild aufzunehmen.')

    # ──────────────────────────────────────────────────────────────────────────
    # Arduino helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_ports(self):
        """Populate the port combobox with currently available serial ports."""
        try:
            from serial.tools import list_ports
            ports = [p.device for p in list_ports.comports()]
        except ImportError:
            ports = []
        self._port_combo["values"] = ports
        current = self._port_var.get()
        if ports and current not in ports:
            self._port_var.set(ports[0])
        elif not ports:
            self._port_var.set("")

    def _toggle_arduino_connection(self):
        with self._arduino_lock:
            if self._arduino is not None:
                self._disconnect_arduino()
            else:
                self._connect_arduino()

    def _connect_arduino(self):
        """Try to open the serial connection (must be called with _arduino_lock held)."""
        try:
            from trash_bin_controller import TrashBinController, find_arduino_port
        except ImportError:
            messagebox.showerror(
                "Importfehler",
                "trash_bin_controller.py nicht gefunden.\n"
                f"Erwartet in: {_CONTROLLER_DIR}"
            )
            return

        port = self._port_var.get() or None

        self._ard_status_lbl.config(text="Verbinde …")
        self._connect_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            ctrl = TrashBinController(port=port)
            self._arduino = ctrl
            
            self._arduino.register_callback("BTN_PRUEFEN", lambda: self.root.after(0, self._on_btn_pruefen_hardware))
            self._arduino.register_callback("BTN_RICHTIG", lambda: self.root.after(0, self._on_btn_richtig_hardware))
            self._arduino.register_callback("BTN_FALSCH", lambda: self.root.after(0, self._on_btn_falsch_hardware))
            
            self._ard_status_dot.config(fg=COLOR_CONNECTED)
            self._ard_status_lbl.config(text=f"Verbunden ({ctrl._serial.name})")
            self._connect_btn.config(text="Trennen", state=tk.NORMAL,
                                     bg="#3d1f1f", activebackground="#5a2b2b")
            self._lid1_btn.config(state=tk.NORMAL)
            self._lid2_btn.config(state=tk.NORMAL)
            self._light_btn.config(state=tk.NORMAL)
        except Exception as exc:
            self._arduino = None
            self._ard_status_dot.config(fg=COLOR_DISCONNECTED)
            self._ard_status_lbl.config(text="Nicht verbunden")
            self._connect_btn.config(state=tk.NORMAL)
            messagebox.showerror("Verbindungsfehler", f"Arduino nicht erreichbar:\n{exc}")

    def _disconnect_arduino(self):
        """Close the serial connection (must be called with _arduino_lock held)."""
        try:
            if self._arduino:
                self._arduino.close()
        except Exception:
            pass
        self._arduino = None
        self._ard_status_dot.config(fg=COLOR_DISCONNECTED)
        self._ard_status_lbl.config(text="Nicht verbunden")
        self._connect_btn.config(text="Verbinden", bg="#1e3a5f",
                                 activebackground="#2a4f80", state=tk.NORMAL)
        self._lid1_btn.config(state=tk.DISABLED)
        self._lid2_btn.config(state=tk.DISABLED)
        # Turn LED off on disconnect (reset state)
        self._led_on = False
        self._light_btn.config(state=tk.DISABLED, text="💡  Licht: AUS",
                               bg="#2a2a3a", fg=FG_MUTED)

    def _send_lid_command(self, label: str):
        """Open the correct lid for *label* if Arduino is connected and checkbox is on.

        After 2 seconds the lid is automatically closed again.
        """
        if not self._send_to_arduino_var.get():
            return
        with self._arduino_lock:
            if self._arduino is None:
                self.status_var.set(
                    self.status_var.get() + "  ⚠ Arduino nicht verbunden"
                )
                return
            lid = _label_to_lid(label)
            try:
                response = self._arduino.open_lid(lid)
                self.status_var.set(
                    self.status_var.get() + f"  |  Arduino: {response}"
                )
            except Exception as exc:
                self.status_var.set(
                    self.status_var.get() + f"  ⚠ Arduino-Fehler: {exc}"
                )
                # Mark as disconnected so the user reconnects
                self._arduino = None
                self.root.after(0, self._refresh_arduino_ui_disconnected)
                return

        # Schedule closing the lid after 2 seconds in a background thread so
        # the serial read inside close_lid() doesn't block the UI.
        def _auto_close():
            time.sleep(2)
            with self._arduino_lock:
                if self._arduino is None:
                    return  # disconnected in the meantime – nothing to do
                try:
                    self._arduino.close_lid(lid)
                except Exception:
                    pass  # best-effort; connection errors are handled elsewhere

        t = threading.Thread(target=_auto_close, daemon=True)
        t.start()

    def _refresh_arduino_ui_disconnected(self):
        self._ard_status_dot.config(fg=COLOR_DISCONNECTED)
        self._ard_status_lbl.config(text="Verbindung verloren")
        self._connect_btn.config(text="Verbinden", bg="#1e3a5f",
                                 activebackground="#2a4f80", state=tk.NORMAL)
        self._lid1_btn.config(state=tk.DISABLED)
        self._lid2_btn.config(state=tk.DISABLED)
        self._led_on = False
        self._light_btn.config(state=tk.DISABLED, text="💡  Licht: AUS",
                               bg="#2a2a3a", fg=FG_MUTED)

    def _toggle_led(self):
        """Toggle the LED strip on/off via Arduino serial command."""
        with self._arduino_lock:
            if self._arduino is None:
                return
            try:
                if self._led_on:
                    self._arduino.led_off()
                    self._led_on = False
                else:
                    self._arduino.led_on()
                    self._led_on = True
            except Exception as exc:
                self._arduino = None
                self.root.after(0, self._refresh_arduino_ui_disconnected)
                messagebox.showerror("Arduino-Fehler", str(exc))
                return
        if self._led_on:
            self._light_btn.config(text="💡  Licht: AN", bg="#4a3a00", fg="#facc15")
        else:
            self._light_btn.config(text="💡  Licht: AUS", bg="#2a2a3a", fg=FG_MUTED)

    def _manual_open_lid(self, lid: int):
        """Manually open *lid* (1 or 2) and close it automatically after 2 seconds.

        Both lid buttons are disabled for the full open→close cycle to prevent
        accidental double-clicks.
        """
        with self._arduino_lock:
            if self._arduino is None:
                return
            try:
                self._arduino.open_lid(lid)
            except Exception as exc:
                self._arduino = None
                self.root.after(0, self._refresh_arduino_ui_disconnected)
                messagebox.showerror("Arduino-Fehler", str(exc))
                return

        # Disable both buttons for the duration of the open→close cycle
        self._lid1_btn.config(state=tk.DISABLED)
        self._lid2_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Deckel {lid} geöffnet – schließt in 2 s …")

        def _auto_close():
            time.sleep(2)
            with self._arduino_lock:
                if self._arduino is None:
                    # Connection lost – UI already updated by disconnect handler
                    return
                try:
                    self._arduino.close_lid(lid)
                    self.root.after(0, lambda: self.status_var.set(f"Deckel {lid} geschlossen."))
                except Exception:
                    pass
            # Re-enable buttons on the main thread regardless of outcome
            self.root.after(0, lambda: (
                self._lid1_btn.config(state=tk.NORMAL),
                self._lid2_btn.config(state=tk.NORMAL),
            ))

        threading.Thread(target=_auto_close, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    # Label rows (built after model is loaded so we know the vocab)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_label_rows(self):
        """Create one progress-bar row per label in the vocabulary."""
        # Remove loading placeholder
        self._loading_lbl.destroy()

        style = ttk.Style()
        style.theme_use("default")

        for idx, label in enumerate(self.vocab):
            color = SLOT_COLORS[idx % len(SLOT_COLORS)]
            bar_style = f"Label{idx}.Horizontal.TProgressbar"
            style.configure(
                bar_style,
                troughcolor=BG_CARD,
                background=color,
                thickness=18,
                borderwidth=0,
            )

            row = tk.Frame(self.results_frame, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=5)

            name_lbl = tk.Label(
                row, text=label, width=12, anchor="w",
                bg=BG_PANEL, fg=FG_TEXT,
                font=("Helvetica", 12, "bold")
            )
            name_lbl.pack(side=tk.LEFT)

            bar_var = tk.DoubleVar(value=0.0)
            bar = ttk.Progressbar(
                row, variable=bar_var,
                maximum=100.0,
                style=bar_style,
                length=160,
            )
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))

            pct_var = tk.StringVar(value="–")
            pct_lbl = tk.Label(
                row, textvariable=pct_var, width=7, anchor="e",
                bg=BG_PANEL, fg=FG_TEXT,
                font=("Helvetica", 12)
            )
            pct_lbl.pack(side=tk.LEFT)

            self._bar_vars[label]   = bar_var
            self._pct_vars[label]   = pct_var
            self._bar_frames[label] = bar

    def _reset_bars(self):
        for label in self.vocab:
            self._bar_vars[label].set(0.0)
            self._pct_vars[label].set("–")

    def _update_bars(self, results: list):
        """Animate bars: fill smoothly over ~400 ms."""
        target = {label: pct for label, pct in results}
        steps  = 20
        delay  = 20  # ms per step

        def step(i):
            frac = (i + 1) / steps
            # ease-out: frac = 1 - (1-t)^2
            ease = 1 - (1 - frac) ** 2
            for label in self.vocab:
                tgt = target.get(label, 0.0)
                self._bar_vars[label].set(tgt * ease)
                if i == steps - 1:
                    self._pct_vars[label].set(f"{tgt:.1f} %")
            if i < steps - 1:
                self.root.after(delay, lambda ni=i+1: step(ni))

        self._reset_bars()
        step(0)

    # ──────────────────────────────────────────────────────────────────────────
    # Camera preview loop
    # ──────────────────────────────────────────────────────────────────────────

    def _update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_bgr = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame_rgb)
                # Scale to fit the video label
                label_w = self.video_label.winfo_width()  or FRAME_WIDTH
                label_h = self.video_label.winfo_height() or FRAME_HEIGHT
                pil.thumbnail((label_w, label_h), Image.LANCZOS)
                self.photo_image = ImageTk.PhotoImage(pil)
                self.video_label.config(image=self.photo_image)
        self.root.after(33, self._update_frame)  # ~30 fps

    # ──────────────────────────────────────────────────────────────────────────
    # "Prüfen" action
    # ──────────────────────────────────────────────────────────────────────────

    def _on_pruefen(self):
        if self.current_frame_bgr is None:
            messagebox.showwarning("Kein Bild", "Noch kein Kamerabild verfügbar.")
            return

        # Reset feedback state & disable feedback buttons
        self._last_capture_path = None
        self._last_top_label    = None
        self.richtig_btn.config(state=tk.DISABLED)
        self.falsch_btn.config(state=tk.DISABLED)

        # Disable button during processing
        self.pruefen_btn.config(state=tk.DISABLED, text="⏳  Analysiere …")
        self.status_var.set("Bild wird gespeichert und analysiert …")
        self.root.update_idletasks()

        # Save the captured frame
        ts        = time.strftime("%Y%m%d_%H%M%S")
        save_path = CAPTURES_DIR / f"capture_{ts}.jpg"
        cv2.imwrite(str(save_path), self.current_frame_bgr)

        # Run prediction
        try:
            results = _predict(self.learner, save_path)
            top_label, top_pct = results[0]
            lid = _label_to_lid(top_label)
            self.status_var.set(
                f"Ergebnis: {top_label}  ({top_pct:.1f} %)  |  Deckel {lid}  |  {save_path.name}"
            )
            self._update_bars(results)

            # Remember for feedback
            self._last_capture_path = save_path
            self._last_top_label    = top_label
            self._last_results      = results
            self.richtig_btn.config(state=tk.NORMAL)
            self.falsch_btn.config(state=tk.NORMAL)

            # Send to Arduino (if enabled)
            self._send_lid_command(top_label)

        except Exception as exc:
            messagebox.showerror("Vorhersagefehler", str(exc))
            self.status_var.set("Fehler bei der Vorhersage.")

        # Re-enable button
        self.pruefen_btn.config(state=tk.NORMAL, text="📸  Prüfen")

    # ──────────────────────────────────────────────────────────────────────────
    # Feedback: Richtig / Falsch
    # ──────────────────────────────────────────────────────────────────────────

    def _save_to_collected_data(self, label: str, src: Path) -> Path:
        """
        Move *src* into  collected-data/<label>/<label>_<timestamp>.jpg
        Returns the destination path.
        """
        dest_dir = COLLECTED_DATA_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Build a safe, unique filename from the original timestamp part
        ts_part = re.sub(r"[^0-9]", "", src.stem)  # keep only digits
        dest = dest_dir / f"{label}_{ts_part or time.strftime('%Y%m%d_%H%M%S')}.jpg"
        shutil.move(str(src), str(dest))
        return dest

    def _on_richtig(self):
        if not self._last_capture_path or not self._last_top_label:
            return
        try:
            dest = self._save_to_collected_data(self._last_top_label, self._last_capture_path)
            self.status_var.set(
                f"✅ Gespeichert als: collected-data/{self._last_top_label}/{dest.name}"
            )
        except Exception as exc:
            messagebox.showerror("Speicherfehler", str(exc))
            return
        # Disable buttons so the same capture can't be saved twice
        self._last_capture_path = None
        self._last_top_label    = None
        self.richtig_btn.config(state=tk.DISABLED)
        self.falsch_btn.config(state=tk.DISABLED)

    def _on_falsch(self):
        if not self._last_capture_path or not self._last_top_label:
            return
        # Open a label-picker dialog
        dialog = LabelPickerDialog(
            self.root,
            vocab=self.vocab,
            wrong_label=self._last_top_label,
        )
        correct_label = dialog.result
        if correct_label is None:
            return  # user cancelled
        try:
            dest = self._save_to_collected_data(correct_label, self._last_capture_path)
            self.status_var.set(
                f"❌→✅ Gespeichert als: collected-data/{correct_label}/{dest.name}"
            )
        except Exception as exc:
            messagebox.showerror("Speicherfehler", str(exc))
            return
        self._last_capture_path = None
        self._last_top_label    = None
        self.richtig_btn.config(state=tk.DISABLED)
        self.falsch_btn.config(state=tk.DISABLED)

    # ── Hardware button handlers ──────────────────────────────────────────────

    def _on_btn_pruefen_hardware(self):
        if str(self.pruefen_btn["state"]) == tk.NORMAL:
            self._on_pruefen()

    def _on_btn_richtig_hardware(self):
        if str(self.richtig_btn["state"]) == tk.NORMAL:
            self._on_richtig()

    def _on_btn_falsch_hardware(self):
        if not self._last_capture_path or not self._last_top_label:
            return
        if not hasattr(self, '_last_results') or len(self._last_results) < 2:
            return
            
        # The other class is the second in the sorted results list
        other_label = self._last_results[1][0]
        
        try:
            dest = self._save_to_collected_data(other_label, self._last_capture_path)
            self.status_var.set(
                f"❌→✅ (HW) Gespeichert als: collected-data/{other_label}/{dest.name}"
            )
        except Exception as exc:
            messagebox.showerror("Speicherfehler", str(exc))
            return
            
        self._last_capture_path = None
        self._last_top_label    = None
        self.richtig_btn.config(state=tk.DISABLED)
        self.falsch_btn.config(state=tk.DISABLED)

    # ──────────────────────────────────────────────────────────────────────────
    # Misc helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _bring_to_front(self):
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def on_close(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        with self._arduino_lock:
            self._disconnect_arduino()
        self.root.quit()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Label-picker dialog  (used by "Falsch" button)
# ──────────────────────────────────────────────────────────────────────────────

class LabelPickerDialog(tk.Toplevel):
    """
    Modal dialog that asks the user to pick the correct label.

    After the dialog closes, inspect ``dialog.result``:
      - str  → the label the user selected
      - None → the user cancelled
    """

    def __init__(self, parent: tk.Tk, vocab: list[str], wrong_label: str):
        super().__init__(parent)
        self.result: str | None = None

        self.title("Richtiges Label wählen")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # ── Header ────────────────────────────────────────────────────────────
        tk.Label(
            self,
            text="❌  Falsche Vorhersage",
            bg=BG_DARK, fg=ACCENT,
            font=("Helvetica", 14, "bold"),
        ).pack(padx=24, pady=(20, 4))

        tk.Label(
            self,
            text=f"KI hat erkannt: \"{wrong_label}\"\nWelches Label ist korrekt?",
            bg=BG_DARK, fg=FG_MUTED,
            font=("Helvetica", 11),
            justify="center",
        ).pack(padx=24, pady=(0, 14))

        # ── Label buttons ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG_DARK)
        btn_frame.pack(padx=24, pady=(0, 8), fill=tk.BOTH, expand=True)

        for idx, label in enumerate(vocab):
            is_wrong = (label == wrong_label)
            color  = SLOT_COLORS[idx % len(SLOT_COLORS)]
            bg_col = BG_CARD if is_wrong else BG_PANEL
            fg_col = FG_MUTED if is_wrong else color
            prefix = "✗  " if is_wrong else "✓  "

            btn = tk.Button(
                btn_frame,
                text=f"{prefix}{label}",
                font=("Helvetica", 13, "bold"),
                bg=bg_col, fg=fg_col,
                activebackground=color, activeforeground=BG_DARK,
                relief=tk.FLAT, bd=0, cursor="hand2",
                padx=14, pady=10,
                command=lambda lbl=label: self._select(lbl),
            )
            btn.pack(fill=tk.X, pady=3)

        # ── Cancel button ─────────────────────────────────────────────────────
        tk.Frame(self, bg=BG_CARD, height=1).pack(fill=tk.X, padx=24, pady=(8, 0))
        tk.Button(
            self,
            text="Abbrechen",
            font=("Helvetica", 11),
            bg=BG_PANEL, fg=FG_MUTED,
            activebackground=BG_CARD, activeforeground=FG_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=10, pady=8,
            command=self._cancel,
        ).pack(fill=tk.X, padx=24, pady=(6, 20))

        # Centre over parent
        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width()  - self.winfo_width())  // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{px}+{py}")

        parent.wait_window(self)

    def _select(self, label: str):
        self.result = label
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():

    root = tk.Tk()
    root.geometry("1020x680")
    root.minsize(800, 560)

    # Set window icon
    try:
        _icon_path = Path(__file__).parent.parent / "image-recognition-icon.png"
        _icon_img = ImageTk.PhotoImage(Image.open(_icon_path))
        root.iconphoto(True, _icon_img)
    except Exception:
        pass  # icon is optional – don't crash if file is missing

    # Dark title bar on macOS (requires Tk ≥ 8.6.12 or Python ≥ 3.11 on arm64)
    try:
        root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "document", "closeBox")
    except Exception:
        pass

    app = PredictGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

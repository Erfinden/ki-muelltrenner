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
import sys
import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH    = Path(__file__).parent / "models" / "trash_classifier.pkl"
CAPTURES_DIR  = Path(__file__).parent / "captures"

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
BTN_TEXT     = "#ffffff"

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

        # ── Arduino state ─────────────────────────────────────────────────────
        self._arduino = None          # TrashBinController instance (or None)
        self._arduino_lock = threading.Lock()

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

        # Camera selector row (shown only when >1 camera found)
        self.cam_row = tk.Frame(left, bg=BG_DARK)
        self.cam_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.camera_var = tk.StringVar()
        self._cam_label = tk.Label(
            self.cam_row, text="Kamera:", bg=BG_DARK, fg=FG_MUTED,
            font=("Helvetica", 10)
        )
        self._cam_combo = ttk.Combobox(
            self.cam_row, textvariable=self.camera_var,
            state="readonly", width=14
        )
        self._cam_combo.bind("<<ComboboxSelected>>", self._switch_camera)

        # "Prüfen" button
        self._build_pruefen_button(left)

        # Status bar
        self.status_var = tk.StringVar(value="Modell wird geladen …")
        status_bar = tk.Label(
            left, textvariable=self.status_var,
            bg=BG_DARK, fg=FG_MUTED, font=("Helvetica", 10),
            anchor="w"
        )
        status_bar.grid(row=3, column=0, sticky="ew", pady=(4, 0))

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
        """Create the big 'Prüfen' button with hover effect."""
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

        self._port_var = tk.StringVar(value="Auto")
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

        # 3. Detect cameras
        self.available_cameras = self._detect_cameras()
        if not self.available_cameras:
            messagebox.showerror("Kamera-Fehler", "Keine Kamera gefunden.")
            self.root.destroy()
            return

        # Show camera selector if multiple found
        if len(self.available_cameras) > 1:
            self._cam_label.pack(side=tk.LEFT)
            self._cam_combo.config(values=[f"Gerät {i}" for i in self.available_cameras])
            self.camera_var.set(f"Gerät {self.available_cameras[0]}")
            self._cam_combo.pack(side=tk.LEFT, padx=(4, 0))

        # 4. Open camera
        self._open_camera(self.available_cameras[0])

        # 5. Ensure captures dir exists
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

        # 6. Enable button & start preview loop
        self.pruefen_btn.config(state=tk.NORMAL)
        self.status_var.set('Bereit \u2013 klicke "Pr\u00fcfen" um ein Bild aufzunehmen.')
        self._update_frame()

    def _detect_cameras(self):
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(i)
                cap.release()
        return cameras if cameras else [0]

    def _open_camera(self, index: int):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            messagebox.showerror("Kamera-Fehler", f"Kamera {index} konnte nicht geöffnet werden.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap = cap
        self.current_camera_index = index

    def _switch_camera(self, event=None):
        selected = self.camera_var.get()
        try:
            new_idx = int(selected.split()[-1])
        except (ValueError, IndexError):
            return
        if new_idx == self.current_camera_index:
            return
        new_cap = cv2.VideoCapture(new_idx)
        if not new_cap.isOpened():
            messagebox.showerror("Kamera-Fehler", f"Kamera {new_idx} nicht verfügbar.")
            self.camera_var.set(f"Gerät {self.current_camera_index}")
            return
        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = new_cap
        self.current_camera_index = new_idx

    # ──────────────────────────────────────────────────────────────────────────
    # Arduino helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_ports(self):
        """Populate the port combobox with currently available serial ports."""
        try:
            from serial.tools import list_ports
            ports = ["Auto"] + [p.device for p in list_ports.comports()]
        except ImportError:
            ports = ["Auto"]
        current = self._port_var.get()
        self._port_combo["values"] = ports
        if current not in ports:
            self._port_var.set("Auto")

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

        port_selection = self._port_var.get()
        port = None if port_selection == "Auto" else port_selection

        self._ard_status_lbl.config(text="Verbinde …")
        self._connect_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            ctrl = TrashBinController(port=port)
            self._arduino = ctrl
            self._ard_status_dot.config(fg=COLOR_CONNECTED)
            self._ard_status_lbl.config(text=f"Verbunden ({ctrl._serial.name})")
            self._connect_btn.config(text="Trennen", state=tk.NORMAL,
                                     bg="#3d1f1f", activebackground="#5a2b2b")
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

    def _send_lid_command(self, label: str):
        """Open the correct lid for *label* if Arduino is connected and checkbox is on."""
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

    def _refresh_arduino_ui_disconnected(self):
        self._ard_status_dot.config(fg=COLOR_DISCONNECTED)
        self._ard_status_lbl.config(text="Verbindung verloren")
        self._connect_btn.config(text="Verbinden", bg="#1e3a5f",
                                 activebackground="#2a4f80", state=tk.NORMAL)

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

            # Send to Arduino (if enabled)
            self._send_lid_command(top_label)

        except Exception as exc:
            messagebox.showerror("Vorhersagefehler", str(exc))
            self.status_var.set("Fehler bei der Vorhersage.")

        # Re-enable button
        self.pruefen_btn.config(state=tk.NORMAL, text="📸  Prüfen")

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
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.geometry("1020x680")
    root.minsize(800, 560)

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

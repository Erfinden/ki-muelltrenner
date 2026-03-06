#!/usr/bin/env python3
"""
train_gui.py
============
KI-Mülltrenner – GUI Training Tool

Lets the user assemble a list of dataset directories (primary + extras),
choose an output model path, and launch training.  Console output from
train.py is streamed into a scrollable log widget; a progress bar advances
as training epochs complete.

Usage:
    python train_gui.py

Dependencies (same venv as the rest of the project):
    fastai, torch, tkinter (stdlib)
"""

import os
import re
import sys
import queue
import subprocess
import threading
import multiprocessing
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────────────

_HERE          = Path(__file__).parent
TRAIN_SCRIPT   = _HERE / "train.py"
DEFAULT_DATA   = _HERE.parent / "fast-labeling" / "dataset"
DEFAULT_MODEL  = _HERE / "models" / "trash_classifier.pkl"
COLLECTED_DATA = _HERE / "collected-data"

# ── Design tokens (matching predict_gui.py) ────────────────────────────────
BG_DARK      = "#1a1a2e"
BG_PANEL     = "#16213e"
BG_CARD      = "#0f3460"
FG_TEXT      = "#e2e8f0"
FG_MUTED     = "#94a3b8"
ACCENT       = "#e94560"
ACCENT_HOVER = "#c73652"
BTN_TEXT     = "#000000"
COLOR_OK     = "#4ade80"
COLOR_WARN   = "#facc15"
COLOR_ERR    = "#f87171"

# Progress bar colours per phase
PHASE_COLORS = ["#60a5fa", "#a78bfa"]  # blue (phase 1), violet (phase 2)

# Total epochs  (must match train.py defaults: EPOCHS_FT=4, EPOCHS_UNF=4)
EPOCHS_PHASE1 = 4
EPOCHS_PHASE2 = 4
TOTAL_EPOCHS  = EPOCHS_PHASE1 + EPOCHS_PHASE2

# Regex to detect an epoch completion line from fastai
# e.g.  "0  0.312  0.271  0.125  00:04"
_EPOCH_LINE_RE = re.compile(r"^\s*(\d+)\s+[\d.]+\s+[\d.]+")


# ──────────────────────────────────────────────────────────────────────────────
# Main application class
# ──────────────────────────────────────────────────────────────────────────────

class TrainGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KI-Mülltrenner – Training")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # Training state
        self._process: subprocess.Popen | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._epochs_done: int = 0
        self._phase: int = 1          # 1 or 2
        self._training_active: bool = False

        # Settings
        self._num_workers_var = tk.IntVar(value=min(4, multiprocessing.cpu_count()))
        self._backbone_var = tk.StringVar(value="timm:tf_efficientnetv2_s")
        self._fp16_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._bring_to_front()

        # Start polling the log queue
        self.root.after(100, self._poll_log_queue)

    # ──────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG_DARK)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        outer.columnconfigure(0, weight=2)
        outer.columnconfigure(1, weight=3)
        outer.rowconfigure(0, weight=1)

        # ── Left column ───────────────────────────────────────────────────
        left = tk.Frame(outer, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)  # dataset list expands

        self._build_dataset_panel(left)
        self._build_model_output_panel(left)
        self._build_config_panel(left)
        self._build_train_button(left)

        # ── Right column ──────────────────────────────────────────────────
        right = tk.Frame(outer, bg=BG_PANEL, padx=14, pady=14)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_progress_panel(right)
        self._build_log_panel(right)

    # ── Dataset panel ─────────────────────────────────────────────────────

    def _build_dataset_panel(self, parent):
        card = tk.Frame(parent, bg=BG_PANEL, padx=14, pady=12)
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(0, weight=1)

        tk.Label(
            card, text="📂  Datensätze", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 13, "bold"), anchor="w"
        ).pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            card, text="Ordner (erster Eintrag = primärer Datensatz):",
            bg=BG_PANEL, fg=FG_MUTED, font=("Helvetica", 10), anchor="w"
        ).pack(fill=tk.X)

        # ── Scrollable card list ──────────────────────────────────────────
        # A Canvas + inner Frame lets each row expand to multiple lines.
        list_outer = tk.Frame(card, bg=BG_CARD, pady=4)
        list_outer.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self._ds_canvas = tk.Canvas(
            list_outer, bg=BG_CARD, highlightthickness=0,
            bd=0, height=160,
        )
        ds_scroll = tk.Scrollbar(
            list_outer, orient=tk.VERTICAL, bg=BG_CARD,
            troughcolor=BG_DARK, activebackground=ACCENT,
            command=self._ds_canvas.yview,
        )
        self._ds_canvas.configure(yscrollcommand=ds_scroll.set)
        ds_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._ds_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._ds_inner = tk.Frame(self._ds_canvas, bg=BG_CARD)
        self._ds_canvas_window = self._ds_canvas.create_window(
            (0, 0), window=self._ds_inner, anchor="nw"
        )

        # Keep inner frame width in sync with canvas width
        def _on_canvas_resize(event):
            self._ds_canvas.itemconfig(self._ds_canvas_window, width=event.width)
            self._refresh_dataset_cards()   # re-wrap labels when width changes
        self._ds_canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region whenever the inner frame changes size
        self._ds_inner.bind(
            "<Configure>",
            lambda e: self._ds_canvas.configure(
                scrollregion=self._ds_canvas.bbox("all")
            )
        )

        # Mousewheel scrolling
        def _on_wheel(event):
            self._ds_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._ds_canvas.bind("<MouseWheel>", _on_wheel)
        self._ds_inner.bind("<MouseWheel>", _on_wheel)

        # Internal list of paths (source of truth)
        self._datasets: list[str] = []
        self._selected_ds_idx: int | None = None

        # Add default datasets if they exist
        if DEFAULT_DATA.exists():
            self._datasets.append(str(DEFAULT_DATA))
        if COLLECTED_DATA.exists():
            self._datasets.append(str(COLLECTED_DATA))
        self._refresh_dataset_cards()

        # Button row
        btn_row = tk.Frame(card, bg=BG_PANEL)
        btn_row.pack(fill=tk.X, pady=(8, 0))

        add_btn = tk.Button(
            btn_row, text="＋  Hinzufügen",
            font=("Helvetica", 11, "bold"),
            bg=BG_CARD, fg=COLOR_OK,
            activebackground=BG_DARK, activeforeground=COLOR_OK,
            relief=tk.FLAT, bd=0, cursor="hand2", padx=10, pady=6,
            command=self._add_dataset,
        )
        add_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        remove_btn = tk.Button(
            btn_row, text="－  Entfernen",
            font=("Helvetica", 11, "bold"),
            bg=BG_CARD, fg=COLOR_ERR,
            activebackground=BG_DARK, activeforeground=COLOR_ERR,
            relief=tk.FLAT, bd=0, cursor="hand2", padx=10, pady=6,
            command=self._remove_dataset,
        )
        remove_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        _bind_hover(add_btn,    BG_DARK, BG_CARD)
        _bind_hover(remove_btn, BG_DARK, BG_CARD)

    def _refresh_dataset_cards(self):
        """Rebuild the card rows from self._datasets."""
        # Destroy all existing cards
        for child in self._ds_inner.winfo_children():
            child.destroy()

        canvas_w = self._ds_canvas.winfo_width()
        wrap_px = max(canvas_w - 80, 120)   # leave room for the index badge

        for idx, path in enumerate(self._datasets):
            is_selected = (idx == self._selected_ds_idx)
            is_primary  = (idx == 0)

            bg = ACCENT if is_selected else BG_PANEL
            fg = BTN_TEXT if is_selected else FG_TEXT
            fg_badge = BTN_TEXT if is_selected else (COLOR_OK if is_primary else FG_MUTED)
            badge_bg = ACCENT_HOVER if is_selected else BG_CARD

            row = tk.Frame(
                self._ds_inner, bg=bg,
                pady=7, padx=8, cursor="hand2",
            )
            row.pack(fill=tk.X, padx=6, pady=(0, 4))

            # Index / role badge
            badge_text = "① Primär" if is_primary else f"  {idx + 1}  "
            tk.Label(
                row, text=badge_text,
                bg=badge_bg, fg=fg_badge,
                font=("Helvetica", 9, "bold"),
                padx=5, pady=2,
            ).pack(side=tk.LEFT, anchor="nw", padx=(0, 8))

            # Path label – wraps to multiple lines
            p = Path(path)
            exists_mark = " ✅" if p.exists() else " ❌"
            path_lbl = tk.Label(
                row, text=path + exists_mark,
                bg=bg, fg=fg,
                font=("Helvetica", 10),
                anchor="w", justify="left",
                wraplength=wrap_px,
            )
            path_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Click anywhere on the row to select it
            def _select(i=idx):
                self._selected_ds_idx = i
                self._refresh_dataset_cards()

            for widget in (row, path_lbl):
                widget.bind("<Button-1>", lambda e, i=idx: _select(i))

            # Hover highlight (only when not selected)
            def _enter(e, r=row, i=idx):
                if i != self._selected_ds_idx:
                    r.config(bg=BG_CARD)
                    for w in r.winfo_children():
                        try: w.config(bg=BG_CARD)
                        except Exception: pass
            def _leave(e, r=row, i=idx):
                if i != self._selected_ds_idx:
                    r.config(bg=BG_PANEL)
                    for w in r.winfo_children():
                        try: w.config(bg=BG_PANEL)
                        except Exception: pass
            row.bind("<Enter>", _enter)
            row.bind("<Leave>", _leave)
            for w in row.winfo_children():
                w.bind("<Enter>", _enter)
                w.bind("<Leave>", _leave)

            # Tooltip on path label
            tip_text = f"{path}\n{'✅ Ordner existiert' if p.exists() else '❌ Ordner nicht gefunden'}"
            Tooltip(path_lbl, text=tip_text)
            Tooltip(row,      text=tip_text)

        # Update scroll region
        self._ds_inner.update_idletasks()
        self._ds_canvas.configure(scrollregion=self._ds_canvas.bbox("all"))

    # ── Model output panel ────────────────────────────────────────────────

    def _build_model_output_panel(self, parent):
        card = tk.Frame(parent, bg=BG_PANEL, padx=14, pady=12)
        card.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        card.columnconfigure(0, weight=1)

        tk.Label(
            card, text="💾  Modell-Ausgabe", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 13, "bold"), anchor="w"
        ).pack(fill=tk.X, pady=(0, 8))

        path_row = tk.Frame(card, bg=BG_PANEL)
        path_row.pack(fill=tk.X)
        path_row.columnconfigure(0, weight=1)

        self._model_path_var = tk.StringVar(value=str(DEFAULT_MODEL))

        path_entry = tk.Entry(
            path_row, textvariable=self._model_path_var,
            bg=BG_CARD, fg=FG_TEXT, insertbackground=FG_TEXT,
            relief=tk.FLAT, bd=0, font=("Helvetica", 10),
        )
        path_entry.grid(row=0, column=0, sticky="ew", ipady=6, padx=(0, 6))

        # Tooltip: show resolved absolute path and whether the dir exists
        def _model_tip_text():
            raw = self._model_path_var.get().strip()
            if not raw:
                return "Kein Pfad angegeben"
            p = Path(raw)
            try:
                resolved = str(p.resolve())
            except Exception:
                resolved = raw
            dir_ok = "✅ Ausgabeordner existiert" if p.parent.exists() else "⚠ Ausgabeordner wird erstellt"
            return f"{resolved}\n{dir_ok}"

        Tooltip(path_entry, text_func=_model_tip_text)

        browse_btn = tk.Button(
            path_row, text="…",
            font=("Helvetica", 11, "bold"),
            bg=BG_CARD, fg=FG_MUTED,
            activebackground=BG_DARK, activeforeground=FG_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2", padx=8, pady=4,
            command=self._browse_model_path,
        )
        browse_btn.grid(row=0, column=1)
        _bind_hover(browse_btn, BG_DARK, BG_CARD)

    # ── Configuration panel ───────────────────────────────────────────────

    def _build_config_panel(self, parent):
        card = tk.Frame(parent, bg=BG_PANEL, padx=14, pady=12)
        card.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        card.columnconfigure(1, weight=1)

        tk.Label(
            card, text="⚙  Konfiguration", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 13, "bold"), anchor="w"
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        # Workers
        tk.Label(card, text="Arbeiter (Workers):", bg=BG_PANEL, fg=FG_MUTED, font=("Helvetica", 10)).grid(row=1, column=0, sticky="w")
        workers_spin = tk.Spinbox(
            card, from_=0, to=32, textvariable=self._num_workers_var,
            bg=BG_CARD, fg=FG_TEXT, buttonbackground=BG_DARK, relief=tk.FLAT, bd=0, width=5
        )
        workers_spin.grid(row=1, column=1, sticky="w", padx=10, pady=2)
        Tooltip(workers_spin, text="Anzahl der CPU-Prozesse zum Bildladen.\nWindows-Tipp: 0 oder 2 testen, falls es stockt.")

        # Backbone
        tk.Label(card, text="Modell-Typ:", bg=BG_PANEL, fg=FG_MUTED, font=("Helvetica", 10)).grid(row=2, column=0, sticky="w")
        backbones = ["timm:tf_efficientnetv2_s", "resnet18", "resnet34", "resnet50", "timm:mobilenetv3_small_100"]
        bb_menu = ttk.OptionMenu(card, self._backbone_var, backbones[0], *backbones)
        bb_menu.grid(row=2, column=1, sticky="ew", padx=10, pady=2)

        # FP16
        tk.Checkbutton(
            card, text="Mixed Precision (FP16)", variable=self._fp16_var,
            bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_DARK, activebackground=BG_PANEL,
            activeforeground=FG_TEXT, relief=tk.FLAT
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

    # ── Start training button ─────────────────────────────────────────────

    def _build_train_button(self, parent):
        self.train_btn = tk.Button(
            parent,
            text="🚀  Training starten",
            font=("Helvetica", 17, "bold"),
            bg=ACCENT, fg=BTN_TEXT,
            activebackground=ACCENT_HOVER, activeforeground=BTN_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2",
            padx=20, pady=14,
            command=self._on_start_training,
        )
        self.train_btn.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        self.train_btn.bind("<Enter>", lambda e: self.train_btn.config(bg=ACCENT_HOVER))
        self.train_btn.bind("<Leave>", lambda e: self.train_btn.config(bg=ACCENT))

        self.status_var = tk.StringVar(value="Bereit.")
        tk.Label(
            parent, textvariable=self.status_var,
            bg=BG_DARK, fg=FG_MUTED, font=("Helvetica", 10), anchor="w"
        ).grid(row=4, column=0, sticky="ew", pady=(6, 0))

    # ── Progress panel ────────────────────────────────────────────────────

    def _build_progress_panel(self, parent):
        pcard = tk.Frame(parent, bg=BG_CARD, padx=14, pady=12)
        pcard.pack(fill=tk.X, pady=(0, 10))
        pcard.columnconfigure(0, weight=1)

        tk.Label(
            pcard, text="Trainingsfortschritt", bg=BG_CARD, fg=FG_TEXT,
            font=("Helvetica", 12, "bold"), anchor="w"
        ).pack(fill=tk.X)

        # Phase label
        self._phase_var = tk.StringVar(value="–")
        tk.Label(
            pcard, textvariable=self._phase_var,
            bg=BG_CARD, fg=FG_MUTED, font=("Helvetica", 10), anchor="w"
        ).pack(fill=tk.X, pady=(2, 6))

        # Overall progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Train.Horizontal.TProgressbar",
            troughcolor=BG_DARK,
            background=PHASE_COLORS[0],
            thickness=20,
            borderwidth=0,
        )
        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_bar = ttk.Progressbar(
            pcard, variable=self._progress_var,
            maximum=100.0, style="Train.Horizontal.TProgressbar",
        )
        self._progress_bar.pack(fill=tk.X)

        # Epoch counter label
        self._epoch_var = tk.StringVar(value="Epoche 0 / 8")
        tk.Label(
            pcard, textvariable=self._epoch_var,
            bg=BG_CARD, fg=FG_MUTED, font=("Helvetica", 10), anchor="e"
        ).pack(fill=tk.X, pady=(4, 0))

    # ── Console log panel ─────────────────────────────────────────────────

    def _build_log_panel(self, parent):
        tk.Label(
            parent, text="Konsolen-Ausgabe", bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 12, "bold"), anchor="w"
        ).pack(fill=tk.X, pady=(0, 6))

        log_frame = tk.Frame(parent, bg=BG_DARK)
        log_frame.pack(fill=tk.BOTH, expand=True)

        scroll = tk.Scrollbar(log_frame, orient=tk.VERTICAL, bg=BG_DARK,
                              troughcolor=BG_DARK, activebackground=ACCENT)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_frame,
            bg=BG_DARK, fg=FG_TEXT,
            font=("Courier", 10),
            relief=tk.FLAT, bd=0,
            state=tk.DISABLED,
            wrap=tk.NONE,
            yscrollcommand=scroll.set,
            insertbackground=FG_TEXT,
            highlightthickness=0,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.log_text.yview)

        # Tag colours for different log categories
        self.log_text.tag_config("ok",    foreground=COLOR_OK)
        self.log_text.tag_config("warn",  foreground=COLOR_WARN)
        self.log_text.tag_config("err",   foreground=COLOR_ERR)
        self.log_text.tag_config("muted", foreground=FG_MUTED)
        self.log_text.tag_config("phase", foreground=PHASE_COLORS[0],
                                 font=("Courier", 10, "bold"))

        # Clear log button
        tk.Button(
            parent, text="Log leeren",
            font=("Helvetica", 10),
            bg=BG_CARD, fg=FG_MUTED,
            activebackground=BG_DARK, activeforeground=FG_TEXT,
            relief=tk.FLAT, bd=0, cursor="hand2", padx=8, pady=4,
            command=self._clear_log,
        ).pack(anchor="e", pady=(6, 0))

    # ──────────────────────────────────────────────────────────────────────
    # Dataset management
    # ──────────────────────────────────────────────────────────────────────

    def _add_dataset(self):
        folder = filedialog.askdirectory(
            title="Datensatz-Ordner wählen",
            initialdir=str(_HERE),
        )
        if not folder:
            return
        if folder not in self._datasets:
            self._datasets.append(folder)
            self._refresh_dataset_cards()
            # Scroll to bottom so the new entry is visible
            self._ds_canvas.after(50, lambda: self._ds_canvas.yview_moveto(1.0))

    def _remove_dataset(self):
        idx = self._selected_ds_idx
        if idx is None or idx >= len(self._datasets):
            messagebox.showinfo(
                "Nichts ausgewählt",
                "Bitte zuerst einen Eintrag durch Klicken auswählen."
            )
            return
        self._datasets.pop(idx)
        # Adjust selection
        if self._datasets:
            self._selected_ds_idx = max(0, idx - 1)
        else:
            self._selected_ds_idx = None
        self._refresh_dataset_cards()

    # ──────────────────────────────────────────────────────────────────────
    # Model path
    # ──────────────────────────────────────────────────────────────────────

    def _browse_model_path(self):
        p = filedialog.asksaveasfilename(
            title="Modell speichern unter …",
            initialdir=str(_HERE / "models"),
            initialfile="trash_classifier.pkl",
            defaultextension=".pkl",
            filetypes=[("FastAI Modell", "*.pkl"), ("Alle Dateien", "*.*")],
        )
        if p:
            self._model_path_var.set(p)

    # ──────────────────────────────────────────────────────────────────────
    # Training launch
    # ──────────────────────────────────────────────────────────────────────

    def _on_start_training(self):
        if self._training_active:
            # Allow cancellation
            self._cancel_training()
            return

        datasets = list(self._datasets)
        if not datasets:
            messagebox.showwarning(
                "Kein Datensatz",
                "Bitte mindestens einen Datensatz-Ordner hinzufügen."
            )
            return

        primary = datasets[0]
        extras  = datasets[1:]

        model_path = Path(self._model_path_var.get().strip())
        if not model_path.suffix:
            model_path = model_path.with_suffix(".pkl")

        # Validate primary dataset
        if not Path(primary).exists():
            messagebox.showerror(
                "Ordner nicht gefunden",
                f"Primärer Datensatz-Ordner nicht gefunden:\n{primary}"
            )
            return

        # Build model dir & name from the output path
        model_dir  = model_path.parent
        model_name = model_path.stem

        # Assemble command
        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--data-dir",      primary,
            "--model-dir",     str(model_dir),
            "--model-name",    model_name,
            "--num-workers",   str(self._num_workers_var.get()),
            "--backbone",      self._backbone_var.get(),
        ]
        if not self._fp16_var.get():
            cmd += ["--no-fp16"]
        if extras:
            cmd += ["--extra-data-dirs"] + extras

        self._reset_progress()
        self._append_log("─" * 60 + "\n", "muted")
        self._append_log(f"Starte Training …\n", "ok")
        self._append_log(f"Befehl: {' '.join(cmd)}\n", "muted")
        self._append_log("─" * 60 + "\n", "muted")

        self._training_active = True
        self.train_btn.config(text="⏹  Abbrechen", bg="#7f1d1d", activebackground="#6b1919")
        self.train_btn.bind("<Enter>", lambda e: self.train_btn.config(bg="#6b1919"))
        self.train_btn.bind("<Leave>", lambda e: self.train_btn.config(bg="#7f1d1d"))
        self.status_var.set("Training läuft …")

        threading.Thread(target=self._run_training, args=(cmd,), daemon=True).start()

    def _run_training(self, cmd: list):
        """Run the training subprocess and feed output to the log queue."""
        try:
            env = os.environ.copy()
            # Prevent Python from buffering stdout so we get live output
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(_HERE),
                env=env,
            )
            for line in self._process.stdout:
                self._log_queue.put(("line", line))
            self._process.wait()
            rc = self._process.returncode
            if rc == 0:
                self._log_queue.put(("done", None))
            else:
                self._log_queue.put(("error", rc))
        except Exception as exc:
            self._log_queue.put(("exception", str(exc)))
        finally:
            self._process = None

    def _cancel_training(self):
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass
        self._append_log("\n⏹ Training abgebrochen.\n", "warn")
        self._finish_training(success=False)

    # ──────────────────────────────────────────────────────────────────────
    # Log queue polling (called periodically on the main thread)
    # ──────────────────────────────────────────────────────────────────────

    def _poll_log_queue(self):
        try:
            while True:
                kind, data = self._log_queue.get_nowait()
                if kind == "line":
                    self._process_log_line(data)
                elif kind == "done":
                    self._append_log("\n✅ Training erfolgreich abgeschlossen!\n", "ok")
                    self._finish_training(success=True)
                elif kind == "error":
                    self._append_log(f"\n❌ Training fehlgeschlagen (Exit-Code {data})\n", "err")
                    self._finish_training(success=False)
                elif kind == "exception":
                    self._append_log(f"\n❌ Fehler beim Starten des Trainingsprozesses:\n{data}\n", "err")
                    self._finish_training(success=False)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_log_queue)

    def _process_log_line(self, line: str):
        """Parse the line, update progress, then append to the log widget."""
        stripped = line.rstrip()

        # Detect phase transitions
        if "Phase 1" in stripped or "backbone frozen" in stripped:
            self._phase = 1
            self._append_log(line, "phase")
            self._update_phase_style()
            return
        if "Phase 2" in stripped or "Fine-tuning full" in stripped or "unfreezing" in stripped.lower():
            self._phase = 2
            self._append_log(line, "phase")
            self._update_phase_style()
            return

        # Detect epoch completion  (fastai table rows: "  0  0.xxx  ...")
        if _EPOCH_LINE_RE.match(stripped):
            self._epochs_done += 1
            pct = min((self._epochs_done / TOTAL_EPOCHS) * 100, 100)
            self._animate_progress(pct)
            self._epoch_var.set(f"Epoche {self._epochs_done} / {TOTAL_EPOCHS}")
            self._append_log(line, "ok")
            return

        # Detect errors / warnings
        low = stripped.lower()
        if any(k in low for k in ("error", "traceback", "exception")):
            self._append_log(line, "err")
            return
        if any(k in low for k in ("warning", "warn")):
            self._append_log(line, "warn")
            return

        # Default
        self._append_log(line, "muted")

    def _update_phase_style(self):
        color = PHASE_COLORS[self._phase - 1]
        style = ttk.Style()
        style.configure("Train.Horizontal.TProgressbar", background=color)
        phase_name = "Phase 1: Kopf-Training (Backbone eingefroren)" \
            if self._phase == 1 else "Phase 2: Vollständiges Fine-Tuning"
        self._phase_var.set(phase_name)

    def _animate_progress(self, target_pct: float):
        """Smoothly animate the progress bar towards *target_pct*."""
        current = self._progress_var.get()
        steps   = 15
        delta   = (target_pct - current) / steps

        def step(i):
            self._progress_var.set(current + delta * (i + 1))
            if i < steps - 1:
                self.root.after(20, lambda ni=i+1: step(ni))

        step(0)

    def _reset_progress(self):
        self._epochs_done = 0
        self._phase       = 1
        self._progress_var.set(0.0)
        self._epoch_var.set(f"Epoche 0 / {TOTAL_EPOCHS}")
        self._phase_var.set("Phase 1: Kopf-Training wird gestartet …")
        self._update_phase_style()

    def _finish_training(self, success: bool):
        self._training_active = False
        self.train_btn.config(text="🚀  Training starten", bg=ACCENT, activebackground=ACCENT_HOVER)
        self.train_btn.bind("<Enter>", lambda e: self.train_btn.config(bg=ACCENT_HOVER))
        self.train_btn.bind("<Leave>", lambda e: self.train_btn.config(bg=ACCENT))
        if success:
            self._animate_progress(100.0)
            self._epoch_var.set(f"Epoche {TOTAL_EPOCHS} / {TOTAL_EPOCHS}")
            self.status_var.set("✅ Training abgeschlossen.")
        else:
            self.status_var.set("⚠ Training gestoppt.")

    # ──────────────────────────────────────────────────────────────────────
    # Log widget helpers
    # ──────────────────────────────────────────────────────────────────────

    def _append_log(self, text: str, tag: str = ""):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ──────────────────────────────────────────────────────────────────────
    # Misc helpers
    # ──────────────────────────────────────────────────────────────────────

    def _bring_to_front(self):
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def on_close(self):
        if self._training_active and self._process:
            if messagebox.askyesno(
                "Training aktiv",
                "Das Training läuft noch. Wirklich beenden?\n"
                "(Der Prozess wird abgebrochen.)"
            ):
                try:
                    self._process.terminate()
                except Exception:
                    pass
            else:
                return
        self.root.quit()
        self.root.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _bind_hover(widget: tk.Button, hover_bg: str, normal_bg: str):
    widget.bind("<Enter>", lambda e: widget.config(bg=hover_bg))
    widget.bind("<Leave>", lambda e: widget.config(bg=normal_bg))


class Tooltip:
    """
    Dark-themed tooltip that appears near the cursor after a short delay.

    Usage (static text)::

        Tooltip(some_widget, text="Fixed message")

    Usage (dynamic text via callable)::

        Tooltip(some_widget, text_func=lambda: compute_text())

    Only one of *text* or *text_func* should be supplied.
    """

    _DELAY_MS  = 600   # ms before the tip appears
    _OFFSET_X  = 16   # px right of cursor
    _OFFSET_Y  = 12   # px below cursor

    def __init__(
        self,
        widget: tk.Widget,
        *,
        text: str = "",
        text_func=None,
    ):
        self._widget    = widget
        self._text      = text
        self._text_func = text_func
        self._tip_win: tk.Toplevel | None = None
        self._after_id  = None
        self._cx = self._cy = 0   # last cursor position (screen coords)

        widget.bind("<Motion>",   self._on_motion,    add="+")
        widget.bind("<Leave>",    self._on_leave,     add="+")
        widget.bind("<Destroy>",  self._on_destroy,   add="+")
        widget.bind("<Button>",   self._on_leave,     add="+")

    # ── event handlers ────────────────────────────────────────────────────

    def _on_motion(self, event):
        self._cx = event.x_root
        self._cy = event.y_root
        self._schedule()

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _on_destroy(self, event=None):
        self._cancel()
        self._hide()

    # ── scheduling ────────────────────────────────────────────────────────

    def _schedule(self):
        self._cancel()
        self._after_id = self._widget.after(self._DELAY_MS, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None

    # ── show / hide ───────────────────────────────────────────────────────

    def _show(self):
        msg = self._text_func() if self._text_func else self._text
        if not msg:
            return

        self._hide()  # destroy any existing tip

        self._tip_win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)   # no window decorations
        tw.wm_attributes("-topmost", True)

        # Outer border frame for a subtle glow effect
        border = tk.Frame(tw, bg=ACCENT, padx=1, pady=1)
        border.pack()

        inner = tk.Frame(border, bg=BG_CARD, padx=10, pady=6)
        inner.pack()

        tk.Label(
            inner,
            text=msg,
            bg=BG_CARD, fg=FG_TEXT,
            font=("Helvetica", 10),
            justify="left",
            wraplength=500,
        ).pack()

        # Position near the cursor
        x = self._cx + self._OFFSET_X
        y = self._cy + self._OFFSET_Y
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self):
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.geometry("1100x720")
    root.minsize(860, 580)

    # Window icon (same as predict_gui.py)
    try:
        from PIL import Image, ImageTk
        _icon_path = Path(__file__).parent.parent / "image-recognition-icon.png"
        _icon_img = ImageTk.PhotoImage(Image.open(_icon_path))
        root.iconphoto(True, _icon_img)
    except Exception:
        pass

    # Dark title bar on macOS
    try:
        root.tk.call("::tk::unsupported::MacWindowStyle", "style", root._w, "document", "closeBox")
    except Exception:
        pass

    app = TrainGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

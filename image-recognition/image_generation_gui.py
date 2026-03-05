#!/usr/bin/env python3
"""
image_generation_gui.py
========================
KI-Mülltrenner – Image Generation / Visualisation Tool

Loads the trained FastAI trash-classifier model and lets the user generate
images via *activation maximisation* (gradient-ascent feature visualisation).
Clicking a class button starts a gradient-ascent loop that, starting from a
random noise image, iteratively modifies pixel values so that the model's
confidence for that class is maximised.  The result is a stylised visualisation
of what the model "thinks" that class looks like.

Dependencies (same venv as image-recognition):
    fastai, torch, torchvision, Pillow, tkinter (stdlib)

Usage:
    python image_generation_gui.py
    python image_generation_gui.py --model ../image-recognition/models/trash_classifier.pkl
"""

import argparse
import io
import random
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageTk

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Visualise trained trash classes via activation maximisation.")
parser.add_argument(
    "--model",
    type=Path,
    default=Path(__file__).parent / "models" / "trash_classifier.pkl",
    help="Path to the exported FastAI model (.pkl). Default: ./models/trash_classifier.pkl",
)
args, _unknown = parser.parse_known_args()

MODEL_PATH: Path = args.model

# ──────────────────────────────────────────────────────────────────────────────
# Theme / palette  (matches predict_gui.py colour scheme)
# ──────────────────────────────────────────────────────────────────────────────

BG_DARK      = "#1a1a2e"
BG_PANEL     = "#16213e"
BG_CARD      = "#0f3460"
FG_TEXT      = "#e2e8f0"
FG_MUTED     = "#94a3b8"
ACCENT       = "#e94560"
ACCENT_HOVER = "#c73652"
BTN_TEXT     = "#ffffff"

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

CANVAS_SIZE = 448   # px – displayed canvas resolution

# ──────────────────────────────────────────────────────────────────────────────
# Activation-maximisation helpers
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE       = 224    # model input resolution (must match training)
STEPS          = 256    # gradient-ascent iterations per generation
LR             = 0.5    # learning rate
TV_WEIGHT      = 1e-3   # total-variation regulariser (light smoothing, not a smear)
L2_WEIGHT      = 1e-6   # L2 regulariser
PREVIEW_EVERY  = 20     # emit a live-preview frame every N gradient steps
BLUR_EVERY     = 64     # very light Gaussian blur only occasionally (prevents grid artefacts)


def _imagenet_stats():
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return mean, std


def _total_variation(t: torch.Tensor) -> torch.Tensor:
    """Anisotropic total-variation regulariser (pixel-to-pixel differences)."""
    dy = (t[:, :, 1:, :] - t[:, :, :-1, :]).abs().mean()
    dx = (t[:, :, :, 1:] - t[:, :, :, :-1]).abs().mean()
    return dy + dx


def _tensor_to_pil(raw: torch.Tensor, size: int) -> Image.Image:
    """Convert an optimisation tensor (1×3×H×W) to a PIL image.

    Maps raw → [0,1] via (tanh(raw) + 1) / 2, which has more uniform gradients
    than sigmoid (avoids near-zero gradients when |raw| > 2).
    Uses .detach().clone() + numpy .copy() to ensure PIL owns its own buffer.
    """
    snapshot = raw.detach().clone()
    img_t = (
        ((torch.tanh(snapshot) + 1.0) / 2.0)   # [0, 1]
        .squeeze(0)                              # [3, H, W]
        .permute(1, 2, 0)                        # [H, W, 3]
        .contiguous()
        .mul(255)
        .byte()
        .cpu()
        .numpy()
        .copy()
    )
    pil = Image.fromarray(img_t, mode="RGB")
    return pil.resize((size, size), Image.LANCZOS)


def generate_class_image(
    full_model: torch.nn.Module,
    class_idx: int,
    seed: int,
    steps: int = STEPS,
    lr: float = LR,
    progress_cb=None,
    preview_cb=None,
) -> Image.Image:
    """
    Gradient-ascent activation maximisation.

    Uses (tanh(raw)+1)/2 as the pixel mapping — better gradient flow than sigmoid.
    Periodically applies a soft Gaussian blur (every BLUR_EVERY steps) to suppress
    grid-like high-frequency artefacts without over-smoothing.

    NOTE: Results depend heavily on model quality.  A model trained on very few images
    (~60) may not have learned strong class-specific visual features, so generated
    images may look like abstract colour/texture blobs rather than recognisable objects.
    More training data → more meaningful visualisations.
    """
    try:
        device = next(full_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    mean, std = _imagenet_stats()
    mean = mean.to(device)
    std  = std.to(device)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    init_noise = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, generator=rng) * 0.3
    raw = torch.nn.Parameter(init_noise.to(device))

    optimizer = torch.optim.Adam([raw], lr=lr)
    full_model.eval()

    print(f"[gen] starting: class={class_idx}, seed={seed}, steps={steps}, lr={lr}, device={device}")

    for step in range(steps):
        optimizer.zero_grad()

        # tanh gives pixel values in [0,1] with stronger gradients than sigmoid
        x      = (torch.tanh(raw) + 1.0) / 2.0   # [0, 1]
        x_norm = (x - mean) / std                  # ImageNet-normalised

        logits = full_model(x_norm)

        loss = -F.log_softmax(logits, dim=1)[0, class_idx]
        loss = loss + TV_WEIGHT * _total_variation(x_norm)
        loss = loss + L2_WEIGHT * (x_norm ** 2).mean()

        loss.backward()

        # ── Diagnostics ───────────────────────────────────────────────────────
        if step == 0 or (step + 1) % PREVIEW_EVERY == 0:
            gn = raw.grad.norm().item() if raw.grad is not None else float("nan")
            print(f"[gen] step {step+1:4d}/{steps}  loss={loss.item():.4f}  "
                  f"grad_norm={gn:.6f}  "
                  f"raw=[{raw.data.min():.3f}, {raw.data.max():.3f}]")

        optimizer.step()

        # Keep raw in a reasonable range so tanh stays out of saturation
        with torch.no_grad():
            raw.data.clamp_(-4.0, 4.0)

        # Light Gaussian blur every BLUR_EVERY steps to suppress grid artefacts
        if (step + 1) % BLUR_EVERY == 0:
            with torch.no_grad():
                blurred = TF.gaussian_blur(
                    raw.data.squeeze(0),
                    kernel_size=5,
                    sigma=0.8,          # subtle – just removes pixel-grid noise
                ).unsqueeze(0)
                raw.data.copy_(blurred)

        if progress_cb:
            progress_cb(step + 1, steps)

        if preview_cb and (step + 1) % PREVIEW_EVERY == 0:
            preview_cb(_tensor_to_pil(raw, CANVAS_SIZE))

    print("[gen] done")
    return _tensor_to_pil(raw, CANVAS_SIZE)


# ──────────────────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────────────────

class ImageGenerationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("KI-Mülltrenner – Bildgenerierung")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # State
        self._learner      = None
        self._vocab: list[str] = []
        self._full_model   = None
        self._generating   = False
        self._photo_image  = None   # final result (kept alive against GC)
        self._photo_preview = None  # live preview frame (kept alive against GC)
        self._current_seed = random.randint(0, 2**31 - 1)

        self._build_ui()
        self._bring_to_front()
        self.root.after(100, self._load_model)

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=BG_DARK)
        title_bar.pack(fill=tk.X, padx=16, pady=(14, 0))

        tk.Label(
            title_bar,
            text="🎨  Bildgenerierung",
            bg=BG_DARK, fg=FG_TEXT,
            font=("Helvetica", 20, "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Label(
            title_bar,
            text="Aktivierungsmaximierung des trainierten Klassifikators",
            bg=BG_DARK, fg=FG_MUTED,
            font=("Helvetica", 11),
            anchor="e",
        ).pack(side=tk.RIGHT, padx=(0, 4))

        # Thin separator
        tk.Frame(self.root, bg=BG_CARD, height=1).pack(fill=tk.X, padx=16, pady=(8, 0))

        # ── Main layout: left canvas  |  right controls ───────────────────────
        outer = tk.Frame(self.root, bg=BG_DARK)
        outer.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=1, minsize=220)
        outer.rowconfigure(0, weight=1)

        # ── Canvas area ───────────────────────────────────────────────────────
        canvas_card = tk.Frame(outer, bg=BG_PANEL, bd=0)
        canvas_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_card.rowconfigure(0, weight=1)
        canvas_card.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_card,
            bg="#0a0a14",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Placeholder text
        self._canvas_placeholder = self.canvas.create_text(
            CANVAS_SIZE // 2, CANVAS_SIZE // 2,
            text="Wähle eine Klasse\num ein Bild zu generieren",
            fill=FG_MUTED,
            font=("Helvetica", 16),
            justify="center",
        )
        self.canvas.config(width=CANVAS_SIZE, height=CANVAS_SIZE)

        # ── Progress bar (below canvas) ───────────────────────────────────────
        prog_frame = tk.Frame(outer, bg=BG_DARK)
        prog_frame.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(6, 0))
        prog_frame.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Gen.Horizontal.TProgressbar",
            troughcolor=BG_PANEL,
            background=ACCENT,
            bordercolor=BG_PANEL,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            prog_frame,
            variable=self._progress_var,
            maximum=STEPS,
            style="Gen.Horizontal.TProgressbar",
            length=CANVAS_SIZE,
        )
        self._progress_bar.grid(row=0, column=0, sticky="ew")

        self._status_var = tk.StringVar(value="Modell wird geladen …")
        tk.Label(
            prog_frame,
            textvariable=self._status_var,
            bg=BG_DARK, fg=FG_MUTED,
            font=("Helvetica", 10),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── Right panel – class buttons ───────────────────────────────────────
        right = tk.Frame(outer, bg=BG_PANEL, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="nsew", rowspan=2)
        right.columnconfigure(0, weight=1)

        tk.Label(
            right,
            text="Klassen",
            bg=BG_PANEL, fg=FG_TEXT,
            font=("Helvetica", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            right,
            text="Klicke um ein Bild zu erzeugen",
            bg=BG_PANEL, fg=FG_MUTED,
            font=("Helvetica", 9),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))

        # Scrollable container for class buttons (in case there are many classes)
        btn_canvas = tk.Canvas(right, bg=BG_PANEL, highlightthickness=0)
        btn_scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=btn_canvas.yview)
        btn_canvas.configure(yscrollcommand=btn_scrollbar.set)

        btn_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        btn_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._btn_frame = tk.Frame(btn_canvas, bg=BG_PANEL)
        self._btn_frame_id = btn_canvas.create_window((0, 0), window=self._btn_frame, anchor="nw")

        def _on_frame_configure(event):
            btn_canvas.configure(scrollregion=btn_canvas.bbox("all"))
        def _on_canvas_configure(event):
            btn_canvas.itemconfig(self._btn_frame_id, width=event.width)

        self._btn_frame.bind("<Configure>", _on_frame_configure)
        btn_canvas.bind("<Configure>", _on_canvas_configure)
        self._btn_canvas = btn_canvas

        # ── Seed & options panel ──────────────────────────────────────────────
        sep = tk.Frame(right, bg=BG_CARD, height=1)
        sep.pack(fill=tk.X, pady=(10, 0))

        opts = tk.Frame(right, bg=BG_PANEL)
        opts.pack(fill=tk.X, pady=(6, 0))
        opts.columnconfigure(1, weight=1)

        tk.Label(
            opts, text="Seed:", bg=BG_PANEL, fg=FG_MUTED,
            font=("Helvetica", 10), anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self._seed_var = tk.StringVar(value=str(self._current_seed))
        seed_entry = tk.Entry(
            opts,
            textvariable=self._seed_var,
            bg=BG_CARD, fg=FG_TEXT,
            insertbackground=FG_TEXT,
            relief=tk.FLAT, bd=4,
            font=("Helvetica", 10),
            width=10,
        )
        seed_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        random_btn = tk.Button(
            opts,
            text="🎲",
            font=("Helvetica", 11),
            bg=BG_CARD, fg=FG_TEXT,
            relief=tk.FLAT, bd=0,
            cursor="hand2", padx=4, pady=2,
            command=self._randomise_seed,
        )
        random_btn.grid(row=0, column=2, padx=(4, 0))

        tk.Label(
            opts, text="Schritte:", bg=BG_PANEL, fg=FG_MUTED,
            font=("Helvetica", 10), anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._steps_var = tk.IntVar(value=STEPS)
        steps_spin = tk.Spinbox(
            opts,
            from_=32, to=1024, increment=32,
            textvariable=self._steps_var,
            bg=BG_CARD, fg=FG_TEXT,
            buttonbackground=BG_CARD,
            insertbackground=FG_TEXT,
            relief=tk.FLAT,
            font=("Helvetica", 10),
            width=6,
        )
        steps_spin.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(8, 0))

        self._loading_lbl_btn = tk.Label(
            self._btn_frame,
            text="Bitte warten…\nModell wird geladen.",
            bg=BG_PANEL, fg=FG_MUTED,
            font=("Helvetica", 11),
            justify="center",
        )
        self._loading_lbl_btn.pack(pady=30)

    def _build_class_buttons(self):
        """Populate the class button area once the vocabulary is known."""
        self._loading_lbl_btn.destroy()

        self._class_btns: dict[str, tk.Button] = {}

        for i, label in enumerate(self._vocab):
            color      = SLOT_COLORS[i % len(SLOT_COLORS)]
            hover_color = self._darken(color, 0.15)

            # 1-px coloured border via a thin Frame wrapper
            frame_wrap = tk.Frame(self._btn_frame, bg=color, padx=1, pady=1)
            frame_wrap.pack(fill=tk.X, pady=(0, 8))

            btn = tk.Button(
                frame_wrap,
                text=f"  {label}",
                font=("Helvetica", 13, "bold"),
                bg=BG_CARD,
                fg=color,
                activebackground=hover_color,
                activeforeground=color,
                relief=tk.FLAT,
                bd=0,
                cursor="hand2",
                padx=10,
                pady=10,
                anchor="w",
                command=lambda lbl=label, idx=i: self._on_generate(lbl, idx),
            )
            btn.pack(fill=tk.BOTH)

            btn.bind("<Enter>", lambda e, b=btn, c=hover_color: b.config(bg=c))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=BG_CARD))

            self._class_btns[label] = btn

    @staticmethod
    def _darken(hex_color: str, factor: float) -> str:
        """Return a darker version of *hex_color* by *factor* (0-1)."""
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Modell nicht gefunden",
                f"Kein Modell unter:\n{MODEL_PATH}\n\n"
                "Bitte zuerst train.py ausführen.",
            )
            self.root.destroy()
            return

        def _worker():
            try:
                from fastai.vision.all import load_learner
                learn = load_learner(MODEL_PATH)
                vocab = list(learn.dls.vocab)
                full_model = learn.model

                self.root.after(0, lambda: self._on_model_loaded(learn, vocab, full_model))
            except Exception as exc:
                # Use default-arg capture (e=exc) so the lambda binds the *value*
                # now – Python 3 sets `exc = None` after the except block exits,
                # which would otherwise make the lambda see None.
                self.root.after(0, lambda e=exc: self._on_model_error(e))

        self._status_var.set("Modell wird geladen …")
        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_loaded(self, learn, vocab, full_model):
        self._learner    = learn
        self._vocab      = vocab
        self._full_model = full_model

        self._status_var.set(f"Modell geladen – {len(vocab)} Klassen gefunden.")
        self._build_class_buttons()

    def _on_model_error(self, exc):
        messagebox.showerror("Ladefehler", f"Modell konnte nicht geladen werden:\n{exc}")
        self.root.destroy()

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_generate(self, label: str, class_idx: int):
        if self._generating:
            return  # busy, ignore

        try:
            seed = int(self._seed_var.get())
        except ValueError:
            seed = random.randint(0, 2**31 - 1)
            self._seed_var.set(str(seed))

        steps = max(32, min(1024, self._steps_var.get()))
        self._progress_var.set(0)
        self._progress_bar.config(maximum=steps)
        self._status_var.set(f"Generiere «{label}» … (Seed {seed})")
        self._generating = True

        # Disable all class buttons while generating
        for btn in self._class_btns.values():
            btn.config(state=tk.DISABLED)

        # Blank the canvas
        self.canvas.delete("all")
        self.canvas.create_text(
            CANVAS_SIZE // 2, CANVAS_SIZE // 2,
            text=f"Generiere «{label}» …",
            fill=FG_MUTED,
            font=("Helvetica", 14),
            justify="center",
        )

        def _progress(step, total):
            self.root.after(0, lambda s=step: self._progress_var.set(s))

        def _preview(pil_img: Image.Image):
            # Called from the worker thread – hand off to the main thread
            self.root.after(0, lambda img=pil_img: self._update_preview(img))

        def _worker():
            try:
                pil_img = generate_class_image(
                    self._full_model,
                    class_idx=class_idx,
                    seed=seed,
                    steps=steps,
                    progress_cb=_progress,
                    preview_cb=_preview,
                )
                self.root.after(0, lambda img=pil_img, lbl=label: self._show_result(img, lbl))
            except Exception as exc:
                import traceback
                traceback.print_exc()   # always print the real error to the terminal
                self.root.after(0, lambda e=exc: self._on_generate_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_preview(self, pil_img: Image.Image):
        """Paint an intermediate preview frame on the canvas (main thread)."""
        # Keep a reference so Tk doesn't garbage-collect the PhotoImage
        self._photo_preview = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.config(width=pil_img.width, height=pil_img.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_preview)

    def _show_result(self, pil_img: Image.Image, label: str):
        self._generating = False

        # Final render – replace preview with the finished image
        self._photo_image   = ImageTk.PhotoImage(pil_img)
        self._photo_preview = None  # release preview reference
        self.canvas.delete("all")
        self.canvas.config(width=pil_img.width, height=pil_img.height)
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

        self._status_var.set(f"Bild für «{label}» generiert.  Seed: {self._seed_var.get()}")
        self._progress_var.set(self._steps_var.get())

        # Re-enable buttons
        for btn in self._class_btns.values():
            btn.config(state=tk.NORMAL)

    def _on_generate_error(self, exc):
        self._generating = False
        messagebox.showerror("Generierungsfehler", f"Fehler bei der Bildgenerierung:\n{exc}")
        self._status_var.set("Fehler aufgetreten.")
        for btn in self._class_btns.values():
            btn.config(state=tk.NORMAL)

    # ──────────────────────────────────────────────────────────────────────────
    # Seed helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _randomise_seed(self):
        new_seed = random.randint(0, 2**31 - 1)
        self._seed_var.set(str(new_seed))

    # ──────────────────────────────────────────────────────────────────────────
    # Window helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _bring_to_front(self):
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGenerationGUI(root)
    root.mainloop()

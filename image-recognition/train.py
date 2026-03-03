"""
KI-Mülltrenner – Image Recognition Training
============================================
Trains a FastAI vision model on trash images to classify categories.

Directory structure expected:
    data/
        Papier/
            img1.jpg
            img2.jpg
            ...
        Plastik/
            ...
        Glas/
            ...
        Metall/
            ...
        Restmuell/
            ...
        Biomuell/
            ...

Run:
    python train.py                          # uses ./data as default
    python train.py --data-dir /path/to/data
    python train.py --data-dir /path/to/data --model-dir /path/to/models --model-name my_model
"""

import argparse

from fastai.vision.all import (
    Path,
    ImageDataLoaders,
    get_image_files,
    parent_label,
    Resize,
    vision_learner,
    resnet34,
    error_rate,
    ClassificationInterpretation,
)
import torch

# ──────────────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Train a FastAI trash-classification model."
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path("data"),
    help="Path to the data folder (one sub-folder per category). Default: ./data",
)
parser.add_argument(
    "--model-dir",
    type=Path,
    default=Path("models"),
    help="Directory where the trained model is saved. Default: ./models",
)
parser.add_argument(
    "--model-name",
    type=str,
    default="trash_classifier",
    help="Base name for the exported model file (without .pkl). Default: trash_classifier",
)
args = parser.parse_args()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DATA_DIR  = args.data_dir
MODEL_DIR = args.model_dir
MODEL_NAME = args.model_name

IMG_SIZE   = 224    # input image size (px)
BATCH_SIZE = 32     # images per mini-batch
EPOCHS_FT  = 4      # fine-tuning epochs (frozen backbone)
EPOCHS_UNF = 4      # epochs after unfreezing the whole network
LEARNING_RATE = 1e-3

# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory '{DATA_DIR}' not found.\n"
        "Please create it and add one sub-folder per trash category with images inside."
    )

MODEL_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
dls = ImageDataLoaders.from_folder(
    DATA_DIR,
    valid_pct=0.2,          # 20 % validation split
    seed=42,
    item_tfms=Resize(IMG_SIZE),
    batch_tfms=None,        # FastAI applies sensible aug defaults
    bs=BATCH_SIZE,
    num_workers=0,          # set >0 on Linux/Mac if you have multiple CPU cores
)

print("Classes:", dls.vocab)
print(f"Training samples : {len(dls.train_ds)}")
print(f"Validation samples: {len(dls.valid_ds)}")

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
learn = vision_learner(dls, resnet34, metrics=error_rate)

# ──────────────────────────────────────────────
# Phase 1 – frozen backbone (transfer learning)
# ──────────────────────────────────────────────
print("\n── Phase 1: Training head (backbone frozen) ──")
learn.fine_tune(EPOCHS_FT, base_lr=LEARNING_RATE)

# ──────────────────────────────────────────────
# Phase 2 – unfreeze & fine-tune whole network
# ──────────────────────────────────────────────
print("\n── Phase 2: Fine-tuning full network ──")
learn.unfreeze()
learn.fit_one_cycle(EPOCHS_UNF, lr_max=slice(1e-6, 1e-4))

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
print("\n── Evaluation on validation set ──")
interp = ClassificationInterpretation.from_learner(learn)
interp.print_classification_report()

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
export_path = (MODEL_DIR / f"{MODEL_NAME}.pkl").resolve()
learn.export(export_path)
print(f"\nModel saved to: {export_path}")

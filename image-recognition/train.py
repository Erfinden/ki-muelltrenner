"""
KI-Mülltrenner – Image Recognition Training
============================================
Trains a FastAI vision model on trash images to classify categories.

Directory structure expected (one sub-folder per category):
    data/
        Papier/   img1.jpg  img2.jpg  ...
        Plastik/  ...
        Glas/     ...
        Metall/   ...
        Restmuell/...
        Biomuell/ ...

Run:
    # Train on original data only (default)
    python train.py

    # Retrain with additional collected-data merged in
    python train.py --extra-data-dirs collected-data

    # Multiple extra directories
    python train.py --data-dir data --extra-data-dirs collected-data extra-data

    # Full options
    python train.py --data-dir /path/to/data \\
                    --extra-data-dirs collected-data \\
                    --model-dir models --model-name my_model
"""

import argparse
from itertools import chain

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
    CategoryBlock,
    ImageBlock,
    DataBlock,
    RandomSplitter,
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
    help="Primary data folder (one sub-folder per category). Default: ./data",
)
parser.add_argument(
    "--extra-data-dirs",
    type=Path,
    nargs="*",
    default=[],
    help=(
        "Additional data folders to merge in (e.g. collected-data). "
        "Images from all directories are pooled before splitting into train/val."
    ),
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
DATA_DIR        = args.data_dir
EXTRA_DATA_DIRS = [Path(p) for p in (args.extra_data_dirs or [])]
MODEL_DIR       = args.model_dir
MODEL_NAME      = args.model_name

IMG_SIZE      = 224    # input image size (px)
BATCH_SIZE    = 32     # images per mini-batch
EPOCHS_FT     = 4      # fine-tuning epochs (frozen backbone)
EPOCHS_UNF    = 4      # epochs after unfreezing the whole network
LEARNING_RATE = 1e-3

# ──────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory '{DATA_DIR}' not found.\n"
        "Please create it and add one sub-folder per trash category with images inside."
    )

for extra in EXTRA_DATA_DIRS:
    if not extra.exists():
        raise FileNotFoundError(
            f"Extra data directory '{extra}' not found.\n"
            "Check the --extra-data-dirs argument."
        )

MODEL_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# Data  –  merge all directories
# ──────────────────────────────────────────────
all_dirs = [DATA_DIR] + EXTRA_DATA_DIRS

if EXTRA_DATA_DIRS:
    print(f"\nMerging {len(all_dirs)} data source(s):")
    for d in all_dirs:
        imgs = get_image_files(d)
        print(f"  {d}  →  {len(imgs)} images")

    # Collect every image file across all directories into a flat list
    all_files = list(chain.from_iterable(get_image_files(d) for d in all_dirs))
    print(f"\nTotal images: {len(all_files)}")

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=lambda _: all_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(IMG_SIZE),
    )
    dls = dblock.dataloaders(DATA_DIR, bs=BATCH_SIZE, num_workers=0)

else:
    # Single directory — same logic as before
    dls = ImageDataLoaders.from_folder(
        DATA_DIR,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=None,
        bs=BATCH_SIZE,
        num_workers=0,
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

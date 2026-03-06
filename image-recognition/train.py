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

import multiprocessing
import timm

from fastai.vision.all import (
    Path,
    ImageDataLoaders,
    get_image_files,
    parent_label,
    Resize,
    aug_transforms,
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
def main():
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
    parser.add_argument(
        "--backbone",
        type=str,
        default="timm:tf_efficientnetv2_s",
        help=(
            "Backbone architecture to use. Supports any fastai built-in (e.g. resnet34, resnet50) "
            "or any timm model name prefixed with 'timm:' "
            "(e.g. timm:tf_efficientnetv2_s, timm:tf_efficientnetv2_m, timm:mobilenetv3_small_100). "
            "Default: resnet34"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, multiprocessing.cpu_count()),
        help="Number of parallel data-loader workers. Default: auto (up to 8).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable mixed-precision (fp16) training for faster GPU training (default: True).",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable mixed-precision training.",
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────
    DATA_DIR        = args.data_dir
    EXTRA_DATA_DIRS = [Path(p) for p in (args.extra_data_dirs or [])]
    MODEL_DIR       = args.model_dir
    MODEL_NAME      = args.model_name
    BACKBONE        = args.backbone
    NUM_WORKERS     = args.num_workers
    USE_FP16        = args.fp16

    IMG_SIZE      = 224    # input image size (px)
    BATCH_SIZE    = 32     # images per mini-batch
    EPOCHS_FT     = 4      # fine-tuning epochs (frozen backbone)
    EPOCHS_UNF    = 4      # epochs after unfreezing the whole network
    LEARNING_RATE = 1e-3

    # Batch augmentations run on the GPU — fast and effective
    BATCH_TFMS    = aug_transforms(mult=1.0)

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

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device  : {device}")
    print(f"Num workers   : {NUM_WORKERS}")
    print(f"Mixed precision: {USE_FP16}")
    print(f"Backbone      : {BACKBONE}")

    # ──────────────────────────────────────────────
    # Data  –  merge all directories
    # ──────────────────────────────────────────────
    all_dirs = [DATA_DIR] + EXTRA_DATA_DIRS

    if EXTRA_DATA_DIRS:
        print(f"\nMerging {len(all_dirs)} data source(s):")
        for d in all_dirs:
            imgs = get_image_files(d)
            print(f"{d}: {len(imgs)} images")

        # Collect every image file across all directories into a flat list
        all_files = list(chain.from_iterable(get_image_files(d) for d in all_dirs))
        print(f"\nTotal images: {len(all_files)}")

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=lambda _: all_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=Resize(IMG_SIZE),
            batch_tfms=BATCH_TFMS,
        )
        dls = dblock.dataloaders(
            DATA_DIR,
            bs=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=(device == "cuda"),
        )

    else:
        # Single directory — same logic as before
        dls = ImageDataLoaders.from_folder(
            DATA_DIR,
            valid_pct=0.2,
            seed=42,
            item_tfms=Resize(IMG_SIZE),
            batch_tfms=BATCH_TFMS,
            bs=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=(device == "cuda"),
        )

    print("Classes:", dls.vocab)
    print(f"Training samples : {len(dls.train_ds)}")
    print(f"Validation samples: {len(dls.valid_ds)}")

    # ──────────────────────────────────────────────
    # Model
    # ──────────────────────────────────────────────
    # Resolve backbone: timm models are passed as strings, fastai built-ins as objects
    _BUILTIN_BACKBONES = {
        "resnet18":  __import__("fastai.vision.all", fromlist=["resnet18"]).resnet18,
        "resnet34":  resnet34,
        "resnet50":  __import__("fastai.vision.all", fromlist=["resnet50"]).resnet50,
        "resnet101": __import__("fastai.vision.all", fromlist=["resnet101"]).resnet101,
    }
    if BACKBONE in _BUILTIN_BACKBONES:
        backbone = _BUILTIN_BACKBONES[BACKBONE]
    elif BACKBONE.startswith("timm:"):
        backbone = BACKBONE[5:]   # fastai accepts the bare timm model name string
    else:
        backbone = BACKBONE       # try passing it directly (e.g. a bare timm name)

    learn = vision_learner(dls, backbone, metrics=error_rate)

    if USE_FP16 and device != "cpu":
        learn = learn.to_fp16()
        print("Mixed precision (fp16) enabled.")

    # ──────────────────────────────────────────────
    # Phase 1 – frozen backbone (transfer learning)
    # ──────────────────────────────────────────────
    print("\nPhase 1: Training head (backbone frozen)")
    learn.fine_tune(EPOCHS_FT, base_lr=LEARNING_RATE)

    # ──────────────────────────────────────────────
    # Phase 2 – unfreeze & fine-tune whole network
    # ──────────────────────────────────────────────
    print("\nPhase 2: Fine-tuning full network")
    learn.unfreeze()
    learn.fit_one_cycle(EPOCHS_UNF, lr_max=slice(1e-6, 1e-4))

    # ──────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────
    print("\nEvaluation on validation set")
    interp = ClassificationInterpretation.from_learner(learn)
    interp.print_classification_report()

    # ──────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────
    export_path = (MODEL_DIR / f"{MODEL_NAME}.pkl").resolve()
    learn.export(export_path)
    print(f"\nModel saved to: {export_path}")

if __name__ == "__main__":
    main()

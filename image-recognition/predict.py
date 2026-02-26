"""
KI-Mülltrenner – Prediction / Inference
========================================
Loads the trained FastAI model and classifies trash images.

Usage (single image):
    python predict.py --image path/to/photo.jpg

Usage (whole folder):
    python predict.py --folder path/to/images/
"""

import argparse
from pathlib import Path

from fastai.vision.all import load_learner, PILImage

MODEL_PATH = Path("models/trash_classifier.pkl")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model '{MODEL_PATH}' not found. Train it first with: python train.py"
        )
    return load_learner(MODEL_PATH)


def predict_image(learn, image_path: Path):
    img = PILImage.create(image_path)
    label, idx, probs = learn.predict(img)
    confidence = probs[idx].item() * 100
    return label, confidence, dict(zip(learn.dls.vocab, (p.item() * 100 for p in probs)))


def main():
    parser = argparse.ArgumentParser(description="Trash image classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=Path, help="Path to a single image")
    group.add_argument("--folder", type=Path, help="Path to a folder of images")
    args = parser.parse_args()

    print(f"Loading model from {MODEL_PATH} …")
    learn = load_model()

    if args.image:
        label, conf, all_probs = predict_image(learn, args.image)
        print(f"\nImage : {args.image.name}")
        print(f"  → Category   : {label}")
        print(f"  → Confidence : {conf:.1f} %")
        print("  → All scores :")
        for cat, pct in sorted(all_probs.items(), key=lambda x: -x[1]):
            print(f"       {cat:<20} {pct:.1f} %")

    elif args.folder:
        image_files = sorted(args.folder.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not image_files:
            print("No images found in folder.")
            return

        print(f"\n{'File':<40} {'Category':<20} {'Confidence':>10}")
        print("-" * 72)
        for img_path in image_files:
            label, conf, _ = predict_image(learn, img_path)
            print(f"{img_path.name:<40} {str(label):<20} {conf:>9.1f} %")


if __name__ == "__main__":
    main()

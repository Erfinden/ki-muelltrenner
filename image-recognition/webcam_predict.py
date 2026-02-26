"""
KI-Mülltrenner – Webcam Prediction (real-time)
===============================================
Opens the webcam, grabs a frame on key-press, and classifies the trash category.

Controls:
    SPACE  – capture and classify the current frame
    Q      – quit

Usage:
    python webcam_predict.py
"""

import cv2
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
from PIL import Image as PILImg
import numpy as np

MODEL_PATH = Path("models/trash_classifier.pkl")

COLORS = [
    (52,  211, 153),   # emerald
    (251, 191,  36),   # amber
    (96,  165, 250),   # blue
    (248, 113, 113),   # red
    (167, 139, 250),   # violet
    (34,  211, 238),   # cyan
]


def bgr_to_pil(frame):
    """Convert an OpenCV BGR frame to a FastAI-compatible PILImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return PILImage.create(PILImg.fromarray(rgb))


def overlay_result(frame, label, confidence, vocab):
    h, w = frame.shape[:2]
    # Semi-transparent background strip at the bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    text = f"{label}  {confidence:.0f}%"
    cv2.putText(frame, text, (20, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SPACE=scan  Q=quit", (20, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model '{MODEL_PATH}' not found. Train it first: python train.py"
        )

    print("Loading model …")
    learn = load_learner(MODEL_PATH)
    print("Model loaded. Opening webcam …")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is it connected?")

    label, confidence = "– press SPACE –", 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        display = overlay_result(frame.copy(), label, confidence, learn.dls.vocab)
        cv2.imshow("KI-Mülltrenner", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            pil_img = bgr_to_pil(frame)
            pred_label, idx, probs = learn.predict(pil_img)
            label = str(pred_label)
            confidence = probs[idx].item() * 100
            print(f"  → {label}  ({confidence:.1f} %)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

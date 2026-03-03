# Image Recognition – KI-Mülltrenner

Trains a FastAI / PyTorch model to classify trash images into categories like **Papier, Plastik, Glas, Metall, Restmüll, Biomüll**.

---

## ⚡ Quick Start

```bash
# 1 – Activate the virtual environment
cd image-recognition
source .venv/bin/activate          # Windows: source .venv/Scripts/activate

# 2 – Install dependencies (only needed once)
pip install -r requirements.txt

# 3 – Add images to the data/ folder (one sub-folder per category)
#     data/Papier/   data/Plastik/   data/Glas/
#     data/Metall/   data/Restmuell/  data/Biomuell/
#     → use the fast-labeling tool from this repo to create images quickly!

# 4 – Train the model (uses ./data by default)
python train.py
#     → saved to models/trash_classifier.pkl

# 4b – Use a custom data folder
python train.py --data-dir /path/to/dataset
#     → saved to models/trash_classifier.pkl

# 4c – Full customisation
python train.py --data-dir /path/to/dataset --model-dir /path/to/models --model-name my_model
#     → saved to /path/to/models/my_model.pkl

# 5a – Classify a single image
python predict.py --image path/to/photo.jpg

# 5b – Classify a whole folder
python predict.py --folder path/to/images/

# 6 – Live webcam (press SPACE to scan, Q to quit)
pip install opencv-python           # only needed once
python webcam_predict.py
```

---

## Folder Structure

```
image-recognition/
├── .venv/                  ← virtual environment (created by you, not committed)
├── data/                   ← training images (one sub-folder per category)
│   ├── Papier/
│   ├── Plastik/
│   ├── Glas/
│   ├── Metall/
│   ├── Restmuell/
│   └── Biomuell/
├── models/                 ← saved model (.pkl) after training
├── train.py                ← trains the model
├── predict.py              ← classify a single image or a folder
├── webcam_predict.py       ← live webcam classification
├── requirements.txt
└── README.md
```

---

## 1 – Create a Virtual Environment

> **macOS / Linux**
```bash
cd image-recognition
python3 -m venv .venv
source .venv/bin/activate
```

> **Windows (Git Bash)**
```bash
cd image-recognition
python -m venv .venv
source .venv/Scripts/activate
```

You should see `(.venv)` at the start of your shell prompt.

---

## 2 – Install Dependencies

```bash
pip install -r requirements.txt
```

This installs **PyTorch**, **FastAI**, Pillow, matplotlib, pandas, scikit-learn, Jupyter, and ipywidgets.

> **GPU (optional):** If you have an NVIDIA GPU, install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> Then run `pip install fastai` afterwards.

---

## 3 – Prepare Training Data

Create a `data/` folder inside `image-recognition/` and put your images in category sub-folders:

```
data/
├── Papier/       ← photos of paper waste
├── Plastik/      ← photos of plastic waste
├── Glas/         ← photos of glass
├── Metall/       ← photos of metal / cans
├── Restmuell/    ← general residual waste
└── Biomuell/     ← organic / food waste
```

**Tips:**
- At least **50–100 images per category** for a decent model; 200+ is better.
- Use the `fast-labeling` tool in this repo to quickly capture and label images with a webcam.
- Images can be `.jpg`, `.jpeg`, or `.png`.

---

## 4 – Train the Model

```bash
# Default – reads images from ./data, saves model to ./models/trash_classifier.pkl
python train.py

# Custom data folder
python train.py --data-dir /path/to/dataset

# Full customisation
python train.py \
  --data-dir /path/to/dataset \
  --model-dir /path/to/models \
  --model-name my_model
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `./data` | Path to the folder containing one sub-folder per category |
| `--model-dir` | `./models` | Directory where the trained model is saved |
| `--model-name` | `trash_classifier` | Base name of the exported `.pkl` file |

The script:
1. Loads images from the given `--data-dir` with an 80/20 train/validation split.
2. Trains a **ResNet-34** backbone (pretrained on ImageNet) with transfer learning.
3. Fine-tunes the full network in a second phase.
4. Prints a classification report.
5. Saves the model to `<model-dir>/<model-name>.pkl`.

---

## 5 – Classify Images

**Single image:**
```bash
python predict.py --image path/to/photo.jpg
```

**Whole folder:**
```bash
python predict.py --folder path/to/images/
```

---

## 6 – Live Webcam Classification

```bash
pip install opencv-python   # only needed once
python webcam_predict.py
```

| Key     | Action                          |
|---------|---------------------------------|
| `SPACE` | Capture frame and classify it   |
| `Q`     | Quit                            |

---

## Model Details

| Setting           | Value          |
|-------------------|----------------|
| Architecture      | ResNet-34      |
| Pretrained on     | ImageNet       |
| Input size        | 224 × 224 px   |
| Batch size        | 32             |
| Phase 1 epochs    | 4 (head only)  |
| Phase 2 epochs    | 4 (full net)   |
| Framework         | FastAI + PyTorch |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the activated venv |
| `FileNotFoundError: data/` | Create the `data/` folder and add images in sub-folders, or pass `--data-dir /your/path` |
| CUDA out of memory | Lower `BATCH_SIZE` in `train.py` |
| Webcam not found | Check camera connection; change `cv2.VideoCapture(0)` index |

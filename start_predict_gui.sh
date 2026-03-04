#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/image-recognition"
source "$SCRIPT_DIR/image-recognition/.venv/bin/activate"
python "$SCRIPT_DIR/image-recognition/predict_gui.py"

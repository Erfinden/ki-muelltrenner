#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/fast-labeling"
source "$SCRIPT_DIR/fast-labeling/venv/bin/activate"
python "$SCRIPT_DIR/fast-labeling/fast-labeling.py"

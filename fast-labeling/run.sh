#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/venv/bin/activate"
python "$SCRIPT_DIR/fast-labeling.py"

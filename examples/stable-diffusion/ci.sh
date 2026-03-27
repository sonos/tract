#!/bin/bash

set -x

VENV_PYTHON=$(dirname $(dirname $(realpath $0)))/stable-diffusion/.venv/bin/python

if [ ! -e .venv ]; then
    # Use virtualenv from api/py venv if system python3 -m venv doesn't work
    if python3 -m venv .venv 2>/dev/null; then
        true
    else
        $(dirname $(dirname $(realpath $0)))/../api/py/.venv/bin/virtualenv .venv
    fi
fi
source .venv/bin/activate

pip install -q torch diffusers "transformers>=4.44,<4.50" accelerate onnxscript onnx Pillow

# Export models to ONNX
python export.py

# Generate reference data
python reference.py

# Run Rust example
cargo run --release

rm -rf assets

#!/bin/bash

set -ex

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Create venv
if [ ! -e .venv ]; then
    if python3 -m venv .venv 2>/dev/null; then
        true
    else
        ../../api/py/.venv/bin/virtualenv .venv
    fi
fi
source .venv/bin/activate

pip install -q torch diffusers transformers accelerate onnxscript onnx Pillow sentencepiece protobuf

# Export models
mkdir -p assets
python export.py

# Run the Rust example
cargo run -p flux-schnell --profile opt-no-lto -- \
    -p "a photo of a cat" -s 4 --seed 42 \
    -o assets/output.png \
    --assets assets

test -f assets/output.png
echo "Done: assets/output.png"

rm -rf assets .venv

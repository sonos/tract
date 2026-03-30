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

# Export models + reference I/O
mkdir -p assets
python export.py
python reference.py

# Validate each model against Python reference
# tract-cli is pre-built by CI, fall back to building locally
TRACT=../../target/opt-no-lto/tract
if [ ! -x "$TRACT" ]; then
    cargo build --profile opt-no-lto -p tract-cli
fi

if nvidia-smi > /dev/null 2>&1; then
    RUNTIME="--cuda"
else
    RUNTIME="-O"
fi

echo "Validating text encoder 1 ($RUNTIME)..."
$TRACT assets/text_encoder.onnx $RUNTIME run \
    --input-from-bundle assets/text_encoder.io.npz \
    --assert-output-bundle assets/text_encoder.io.npz --approx very

echo "Validating text encoder 2 ($RUNTIME)..."
$TRACT assets/text_encoder_2.onnx $RUNTIME run \
    --input-from-bundle assets/text_encoder_2.io.npz \
    --assert-output-bundle assets/text_encoder_2.io.npz --approx very

echo "Validating transformer ($RUNTIME)..."
$TRACT assets/transformer.onnx $RUNTIME run \
    --input-from-bundle assets/transformer.io.npz \
    --assert-output-bundle assets/transformer.io.npz --approx very

echo "Validating VAE decoder ($RUNTIME)..."
$TRACT assets/vae_decoder.onnx $RUNTIME run \
    --input-from-bundle assets/vae_decoder.io.npz \
    --assert-output-bundle assets/vae_decoder.io.npz --approx very

# Run the Rust example
cargo run -p stable-diffusion-3 --profile opt-no-lto -- \
    -p "a photo of a cat" -s 10 --seed 42 \
    -o assets/test_output.png \
    --assets assets

test -f assets/test_output.png
echo "CI passed: test_output.png generated"

rm -rf assets .venv

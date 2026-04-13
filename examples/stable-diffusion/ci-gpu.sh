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

pip install -q torch diffusers "transformers>=4.44,<4.50" accelerate onnxscript onnx Pillow

# Export models to ONNX
mkdir -p assets
python export.py

# Generate reference I/O data for validation
python reference.py

# Validate each model against Python reference
# tract-cli is pre-built by CI, fall back to building locally
TRACT=../../target/opt-no-lto/tract
if [ ! -x "$TRACT" ]; then
    cargo build --profile opt-no-lto -p tract-cli
fi

if nvidia-smi > /dev/null 2>&1; then
    RUNTIME="--cuda"
    GPU_ASSERT="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,IsNan,Gather*,Reduce*,Cast"
elif [ "$(uname)" = "Darwin" ] && system_profiler SPDisplaysDataType 2>/dev/null | grep -qi metal; then
    RUNTIME="--metal"
    GPU_ASSERT="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,IsNan,Gather*,Reduce*,Cast"
else
    RUNTIME="-O"
    GPU_ASSERT=""
fi

echo "Validating text encoder ($RUNTIME)..."
$TRACT assets/text_encoder.onnx $RUNTIME run \
    --input-from-bundle assets/text_encoder.io.npz \
    --assert-output-bundle assets/text_encoder.io.npz --approx very $GPU_ASSERT

echo "Validating VAE decoder ($RUNTIME)..."
$TRACT assets/vae_decoder.onnx $RUNTIME run \
    --input-from-bundle assets/vae_decoder.io.npz \
    --assert-output-bundle assets/vae_decoder.io.npz --approx very $GPU_ASSERT

echo "Validating UNet ($RUNTIME)..."
$TRACT assets/unet.onnx $RUNTIME run \
    --input-from-bundle assets/unet.io.npz \
    --assert-output-bundle assets/unet.io.npz --approx very $GPU_ASSERT

# Run the Rust example
cargo run -p stable-diffusion --profile opt-no-lto -- \
    -p "a photo of a cat" -s 10 --seed 42 \
    -o assets/test_output.png \
    --assets assets

test -f assets/test_output.png
echo "CI passed: test_output.png generated"

rm -rf assets .venv

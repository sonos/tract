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

pip install -q torch diffusers transformers accelerate onnxscript onnx Pillow

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
    GPU_ASSERT="--assert-op-only Cuda*,Gpu*,DeviceSync*,Const,Source,IsNan,Gather*,Reduce*"
elif [ "$(uname)" = "Darwin" ] && system_profiler SPDisplaysDataType 2>/dev/null | grep -qi metal; then
    RUNTIME="--metal"
    GPU_ASSERT="--assert-op-only Metal*,Gpu*,DeviceSync*,Const,Source,IsNan,Gather*,Reduce*"
else
    RUNTIME="-O"
    GPU_ASSERT=""
fi

echo "Validating text encoder 1 ($RUNTIME)..."
$TRACT assets/text_encoder.onnx $RUNTIME run \
    --input-from-bundle assets/text_encoder.io.npz \
    --assert-output-bundle assets/text_encoder.io.npz --approx very $GPU_ASSERT

echo "Validating text encoder 2 ($RUNTIME)..."
$TRACT assets/text_encoder_2.onnx $RUNTIME run \
    --input-from-bundle assets/text_encoder_2.io.npz \
    --assert-output-bundle assets/text_encoder_2.io.npz --approx very $GPU_ASSERT

# Validate UNet — needs >=16GB VRAM for f32, skip on smaller GPUs
if nvidia-smi > /dev/null 2>&1; then
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$GPU_MEM_MB" -ge 16000 ] 2>/dev/null; then
        echo "Validating UNet f32 ($RUNTIME)..."
        $TRACT assets/unet.onnx $RUNTIME run \
            --input-from-bundle assets/unet.io.npz \
            --assert-output-bundle assets/unet.io.npz --approx very $GPU_ASSERT
    else
        echo "Skipping UNet validation (GPU has ${GPU_MEM_MB}MB, need >=16000MB for f32)"
    fi
else
    echo "Validating UNet f32 (CPU)..."
    $TRACT assets/unet.onnx -O run \
        --input-from-bundle assets/unet.io.npz \
        --assert-output-bundle assets/unet.io.npz --approx very
fi

echo "Validating VAE decoder ($RUNTIME)..."
$TRACT assets/vae_decoder.onnx $RUNTIME run \
    --input-from-bundle assets/vae_decoder.io.npz \
    --assert-output-bundle assets/vae_decoder.io.npz --approx very $GPU_ASSERT

# Run the Rust example
cargo run -p stable-diffusion-xl --profile opt-no-lto -- \
    -p "a photo of a cat" -s 10 --seed 42 \
    -o assets/test_output.png \
    --assets assets

test -f assets/test_output.png
echo "CI passed: test_output.png generated"

rm -rf assets .venv

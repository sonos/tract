#!/bin/bash

set -ex

[ -e .venv ] || python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "torch_to_nnef>=0.24.12" "torch_to_nnef_llm>=0.24.12" qwen-asr

mkdir -p assets
wget -qN https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O assets/2086-149220-0033.wav
rm -rf assets/model
python export.py assets/model assets/2086-149220-0033.wav

cargo run --release
rm -rf assets

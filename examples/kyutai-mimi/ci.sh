#!/bin/bash

set -ex

[ -e .venv ] || python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "torch_to_nnef>=0.24.12" transformers soundfile

# The export checks the graphs against the in-tree tract: replicate padding
# needs a newer NNEF loader than torch_to_nnef's pinned release.
cargo build --release -p tract-cli
export TRACT_BIN=$(realpath ../../target/release/tract)

mkdir -p assets
wget -qN https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O assets/2086-149220-0033.wav
rm -rf assets/model assets/reference
python export.py assets assets/2086-149220-0033.wav

cargo run --release
rm -rf assets

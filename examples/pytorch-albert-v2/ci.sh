#!/bin/sh

set -e
sudo apt-get install -y python3-virtualenv
virtualenv venv
. ./venv/bin/activate
pip install -q torch transformers onnx accelerate
python export.py
cargo run --release
rm -rf venv
cargo clean

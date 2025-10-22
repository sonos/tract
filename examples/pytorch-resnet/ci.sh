#!/bin/sh

set -e
sudo apt-get install -y python3-virtualenv
virtualenv venv
. ./venv/bin/activate
pip install -q torch torchvision onnx onnxscript
python export.py
cargo run
rm -rf venv

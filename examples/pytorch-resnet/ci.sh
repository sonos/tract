#!/bin/sh

set -ex

virtualenv venv
. ./venv/bin/activate
pip install torch torchvision
python export.py
cargo run

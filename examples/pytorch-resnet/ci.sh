#!/bin/sh

set -ex
sudo apt-get install -y python3-virtualenv
virtualenv venv
. ./venv/bin/activate
pip install torch torchvision
python export.py
cargo run

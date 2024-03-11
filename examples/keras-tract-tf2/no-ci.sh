#!/bin/sh

set -e

sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 python3-virtualenv

virtualenv venv
. venv/bin/activate

pip install -r requirements.txt
python example.py
cargo run

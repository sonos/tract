#!/bin/sh

set -e


sudo apt-get update

# Install required libraries
# sudo apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 python3 python3-venv
sudo apt-get install -y libxrandr2 libxss1 libxcursor1 libxcomposite1 libxi6 libxtst6 python3 python3-venv

python3 -m venv venv
. venv/bin/activate


pip install -r requirements.txt
python example.py
cargo run

#!/bin/bash

set -x

[ -e .venv ] || python3 -m venv .venv
source .venv/bin/activate

 pip install  "nemo-toolkit[asr]" "torch_to_nnef[nemo_tract]" 

mkdir -p assets
wget -qN https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O  assets/2086-149220-0033.wav
rm -rf assets/model
t2n_export_nemo -s nvidia/parakeet-tdt-0.6b-v3 -e assets/model -tt skip --split-joint-decoder

cargo run --release
rm -rf assets

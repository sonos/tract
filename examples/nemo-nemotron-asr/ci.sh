#!/bin/bash

set -x

[ -e .venv ] || python3 -m venv .venv
source .venv/bin/activate

pip install "nemo-toolkit[asr]" "torch_to_nnef[nemo_tract]"

mkdir -p assets
wget -qN https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O assets/2086-149220-0033.wav
rm -rf assets/model
t2n_export_nemo -s nvidia/nemotron-speech-streaming-en-0.6b -e assets/model -tt skip --split-joint-decoder

# Inject missing upper bound assertion into encoder model (~6.7min at 100Hz)
enc_tgz=assets/model/encoder.nnef.tgz
tmpdir=$(mktemp -d)
tar xzf "$enc_tgz" -C "$tmpdir"
sed -i '/^extension tract_symbol AUDIO_SIGNAL__TIME;/a extension tract_assert AUDIO_SIGNAL__TIME<=39993;' "$tmpdir/graph.nnef"
tar czf "$enc_tgz" -C "$tmpdir" .
rm -rf "$tmpdir"

cargo run --release
rm -rf assets

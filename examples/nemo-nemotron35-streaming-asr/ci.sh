#!/bin/bash

set -x

[ -e .venv ] || python3 -m venv .venv
source .venv/bin/activate

# The multilingual "WithPrompt" RNNT model class (rnnt_bpe_models_prompt) is not
# in any released nemo-toolkit yet (2.7.x ships only the hybrid variant), so NeMo
# must come from git main. torch_to_nnef >= 0.24 provides the prompt-head export.
pip install "torch_to_nnef[nemo_tract]>=0.24"
pip install Cython packaging
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

mkdir -p assets
wget -qN https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O assets/2086-149220-0033.wav
rm -rf assets/model

# --fuse-prompt-into-encoder folds the language head into the encoder, which then
# takes a `lang_id` input. t2n bakes the AUDIO_SIGNAL__TIME<=39993 assertion in
# for this model, so no manual graph patching is needed (unlike the en example).
t2n_export_nemo \
    -s nvidia/nemotron-3.5-asr-streaming-0.6b \
    -e assets/model \
    -tt skip \
    --split-joint-decoder \
    --fuse-prompt-into-encoder

cargo run --release
rm -rf assets

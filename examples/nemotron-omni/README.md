# Nemotron 3 Nano Omni

[Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
is NVIDIA's multimodal reasoning model: it reads text, images and audio, and answers in text.
Three networks run in tract here:

- **The language model**, Nemotron-H 30B-A3B: a hybrid of 23 Mamba-2 layers, 23
  mixture-of-experts layers (128 experts, 6 active per token) and 6 attention layers. It
  activates about 3B of its 31B parameters per token.
- **The vision encoder**, C-RADIOv4-H with a projector: one image at its own aspect ratio, up to
  13312 patches of 16x16 pixels, to one embedding per 2x2 patches.
- **The audio encoder**, a Parakeet FastConformer with a projector: one clip of 16 kHz audio to
  one embedding per 80 ms. Its mel spectrogram runs in tract too.

The example encodes an image, an audio clip or both, splices their embeddings into the prompt,
then decodes the answer greedily.

## Exporting the model

```sh
python3 -m venv venv && source venv/bin/activate
pip install torch transformers huggingface_hub \
    "torch_to_nnef @ git+https://github.com/sonos/torch-to-nnef" \
    "torch_to_nnef_llm @ git+https://github.com/sonos/torch-to-nnef#subdirectory=packages/llm"
python export.py assets
```

It downloads the checkpoint (about 62 GB), then writes about 23 GB of graphs to `assets`: the
language model in f16, its routed experts quantized to Q4_0. The export needs about 58 GB of
RAM and takes about 15 minutes, plus room on disk next to `assets` for a temporary copy of the
weights.
`--checkpoint DIR` exports from a copy of the checkpoint already on disk.

## Running it

```sh
cargo run --release --features cuda -- --image photo.jpg "Describe this image."
cargo run --release --features cuda -- --audio speech.wav "Transcribe this audio."
cargo run --release --features cuda -- --image chart.png --audio question.wav "Answer the question about the chart."
```

`--think` lets the model reason in a `<think>` block before it answers, which it prints too.
WAV files can be at any sample rate; the first channel is used.

The model takes about 23 GB of GPU memory. On an RTX 5090, loading takes about 6 s, a
300-token image prompt 0.2 s, and the answer comes at about 150 tokens per second.

Images are decoded by the `image` crate, and JPEG decoders do not agree to the last bit: on a
JPEG, the image embeddings differ from NVIDIA's processor a little more than on a PNG, with no
visible effect on the answers.

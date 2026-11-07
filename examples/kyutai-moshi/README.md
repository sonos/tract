# Kyutai Moshi

[Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) is Kyutai's full-duplex
speech dialogue model: it listens and speaks at the same time, one 80 ms frame at a time.
It is three networks:

- **Mimi**, the audio codec of the [kyutai-mimi](../kyutai-mimi) example: 24 kHz audio to
  8 codes per frame, and back.
- **Helium**, a 7B transformer: each frame it reads 17 token streams (Moshi's text, Moshi's 8
  audio codes, the user's 8 audio codes) and predicts Moshi's next text token.
- **The depformer**, a small transformer that runs 8 steps per frame, each with its own
  weights, to produce Moshi's 8 audio codes.

This example runs all three frame by frame on Kyutai's token schedule (Moshi's acoustic
codes run one frame behind its semantic code) with Kyutai's sampling settings, from a WAV
of the user's side to a WAV of Moshi's voice.

## Running it

```sh
pip install torch transformers "torch_to_nnef>=0.24.12" sentencepiece soundfile huggingface_hub
python export.py assets sample.wav
cargo run --release --features cuda -- --input user.wav --output moshi.wav
```

On Apple GPUs, build with `--features metal` and pass `--runtime metal`. Input WAVs are
24 kHz mono. Moshi's words are printed as they come, its voice goes to the output WAV.

`export.py` downloads the checkpoint (about 15 GB) and writes about 16 GB of graphs. The
Mimi export checks its graphs with a tract binary that reads replicate padding: build
`tract-cli` from this repository and point `TRACT_BIN` at it, as the kyutai-mimi example's
`ci.sh` does.

## What it needs

Helium is exported in f16 and reads all of its 14 GB of weights every frame, so the GPU
needs about 17 GB of memory, and keeping up with 80 ms frames takes well over 175 GB/s of
memory bandwidth, plus the time of the depformer's 8 sequential steps and of Mimi. The
example prints the time per frame against that 80 ms budget.

A conversation can last up to Helium's 3000-frame context (4 minutes); past that the KV cache
would need a sliding window, which this example does not do.

## Checking against Kyutai

```sh
pip install --no-deps moshi && pip install einops sphn
python export.py assets sample.wav --check
cargo run --release --features cuda -- --check
```

`--check` makes `export.py` run Kyutai's own `LMGen` greedily over the first 20 frames of the
sample (in f32 on the CPU: plan for 60 GB of RAM), and the example replay the same user codes
greedily and compare every token Moshi emits. They match exactly. `moshi` pins an older
torch, hence `--no-deps`.

## How the graphs are cut

- **Helium** is one step graph: past keys and values come in as inputs and leave
  concatenated with the new ones, which `transformers_detect_all` folds into a stateful KV
  cache. Moshi's rotary embedding rotates interleaved pairs; the export permutes the query
  and key rows of each head to the half-split layout, which leaves attention unchanged and
  gives the form tract fuses.
- **The depformer** is one graph per step: each step has its own weights, and its past is
  always the earlier steps of the frame, so every shape is fixed.
- **Mimi** is pulsed to one frame, as in the kyutai-mimi example.

The checkpoint is Kyutai's; see its model card for its licence.

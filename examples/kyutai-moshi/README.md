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

## Talking to it live

Moshi needs a large GPU, while the microphone and speakers are often on another machine. The
example splits the two over TCP: `moshi-server` runs the model, `moshi-client` streams the
microphone in 80 ms frames and plays Moshi's voice as it comes back.

On the GPU machine:

```sh
cargo run --release --features cuda --bin moshi-server
```

It loads `assets/model` (`--q4` as above), listens on port 8998 (`--listen` to change it)
and holds one conversation at a time, each starting fresh. On the machine with the microphone:

```sh
cargo run --release --features live --bin moshi-client -- --server <gpu-host>:8998
```

The client uses the default input and output devices and resamples to and from 24 kHz.
Use headphones: through speakers, Moshi hears itself. Stop with Ctrl-C. Without the `live`
feature, `--input` and `--output` make the client stream a WAV in real time and record the
answer instead.

The protocol is bare PCM over a trusted network, with no authentication or encryption.

## What it needs

Helium is exported in f16 and reads all of its 14 GB of weights every frame, so the GPU
needs about 17 GB of memory, and keeping up with 80 ms frames takes well over 175 GB/s of
memory bandwidth, plus the time of the depformer's and Mimi's kernels. The example prints
the time per frame against that 80 ms budget.

`--q4` quantizes Helium and the depformer to Q4_0 blocks when loading them: about 5 GB of
memory and a quarter of the bandwidth, for GPUs short of either. It costs some quality, and
the reads of the KV cache still grow with the conversation.

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
- **The depformer** is one graph for a whole frame: its 8 steps, each with its own weights,
  chained with their sampling in between. Every token is drawn like Kyutai's `sample_token`
  (top-k, then softmax at the temperature) as the argmax of its top-k logits plus Gumbel
  noise the host passes in; zero noise is greedy. The text token is sampled there too, from
  Helium's logits, so a frame reads back nothing but its 9 tokens.
- **Mimi** is pulsed to one frame, as in the kyutai-mimi example.

The checkpoint is Kyutai's; see its model card for its licence.

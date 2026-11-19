# Voxtral Mini 4B Realtime

[Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) is
Mistral's streaming speech recognition model (Apache-2.0, 13 languages). It was trained to
stream: it emits one text token per 80 ms of audio, a fixed delay behind it (480 ms by
default), and never revisits what it wrote. It is two networks:

- **An audio encoder** (about 1B parameters): a log-mel front-end, two causal convolutions,
  and 32 transformer layers whose attention is causal over a 750-frame (15 s) sliding window.
  A projector turns every 4 of its 50 Hz frames into one 3072-wide audio embedding.
- **A decoder** (a 3.4B Ministral language model): each step it reads the sum of the
  embedding of its previous token and of the audio embedding for that step, and predicts the
  next token. Most tokens are padding; `[STREAMING_WORD]` marks the start of each word.

This example runs both one 80 ms chunk at a time and prints the transcript as it comes.

## Running it

```sh
pip install torch transformers "mistral-common[audio]" torch_to_nnef soundfile huggingface_hub
python export.py assets
cargo run --release --features cuda -- --runtime cuda --f16 [file.wav]
```

Use `--features metal` or `--features wgpu` and the matching `--runtime` for other GPUs. Input
WAVs are 16 kHz mono; without one, the example transcribes the LibriSpeech sample `export.py`
downloads.

`export.py` downloads the checkpoint (8.9 GB) and writes about 18 GB of f32 graphs, in about 20
minutes. `--delay-ms` exports for another transcription delay: a multiple
of 80 ms from 80 to 1200, or 2400. Shorter delays answer sooner and transcribe worse.

`--f16` runs both graphs in f16 (the log-mel stays in f32), `--q4` quantizes the decoder's
weights to Q4_0 blocks, which costs some accuracy. The example prints the time per step against
the 80 ms of audio it covers. On an RTX 5090 a step takes about 16 ms in f32 and 10 ms in f16
over CUDA, and about 42 ms in f32 over wgpu (Vulkan).

## Checking against transformers

```sh
python export.py assets --check
cargo run --release --features cuda -- --runtime cuda --check
```

`--check` makes `export.py` run the transformers implementation greedily over the sample (in
f32 on the CPU, about 20 GB of RAM), and the example compare every token it emits with it. They
match exactly, in f32 and f16.

## How the graphs are cut

- **The encoder** is written as one pass over a whole recording, and tract's pulse
  transformation turns it into a stateful graph that takes 1280 samples (80 ms) and returns one
  audio embedding, with no delay: the causal convolutions keep their last inputs and the
  sliding-window attention its last 750 keys and values. The log-mel is in the graph as a
  convolution by a windowed Fourier basis. Each mel frame reads 40 samples past its hop, so the
  host feeds the encoder its input 40 samples (2.5 ms) early, as the reference does.
- **The decoder** is one step graph: past keys and values come in as inputs and leave
  concatenated with the new ones, which `transformers_detect_all` folds into a stateful KV
  cache. The model conditions its norms on the transcription delay; the export folds that in,
  so each delay is its own export.
- **The host** feeds the stream the way the reference does: 32 tokens (2.56 s) of silence first,
  during which the decoder reads its prompt (a BOS, then a padding token per token of silence and
  of delay), then the audio, then enough silence for the decoder to catch up with its end.

The decoder attends over its whole past. The model was trained with an 8192-token (11-minute)
sliding window there, which this example does not apply: past 11 minutes of audio, the decoder
attends further back than it was trained to.

The checkpoint is Mistral's; see its model card for its licence.

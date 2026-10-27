# Qwen3-ASR 0.6B

Speech recognition with [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
(Apache-2.0, 30 languages, built-in language identification), exported to NNEF and run by
tract.

```
./ci.sh
```

exports the model, then runs both binaries on a LibriSpeech sample and checks their
transcripts. Once `assets/` exists:

```
cargo run --release --bin batch [file.wav]
cargo run --release --bin stream [file.wav]
```

Input is 16 kHz mono 16-bit PCM. Add `--features cuda` or `--features metal` to run the encoder
and decoder on a GPU.

## Model

`export.py` writes three graphs and the tokenizer to `assets/model/`:

- `preprocessor.nnef.tgz`: log10 mel spectrogram (Whisper front-end, 128 bins, 10 ms hop).
  The host applies Whisper's centering, its floor at the maximum minus 8 and its scaling.
- `encoder.nnef.tgz`: the audio encoder over one attention window (see below), up to 8 chunks
  of 100 frames in, 13 audio embeddings per chunk out.
- `decoder/`: a Qwen3 language model with its KV cache. The audio embeddings are spliced in at
  the `<|audio_pad|>` positions of the prompt.

The encoder downsamples each second of audio on its own through three strided convolutions, then
runs 18 transformer layers whose attention is confined to blocks of 8 seconds: tokens attend to
every token of their block, past and future, and to nothing outside it. Blocks are independent,
so the host encodes long audio one 8 s window at a time.

The decoder is prompted with a chat turn holding the audio, and answers
`language English<asr_text>...`.

## Batch

`batch` computes the features of the whole file, encodes each window, and decodes greedily.
Its transcripts match the reference PyTorch implementation.

## Streaming

`stream` feeds the audio by 2 s steps and prints a transcript after each.

This model is a different problem from the Nemotron streaming models
(`../nemo-nemotron*-streaming-asr`). Those were trained to stream: their attention only sees a
bounded past and a short lookahead, so tract's pulse transformation turns them into stateful
graphs that process each frame once, with a fixed latency.

Qwen3-ASR was not. Two things stand in the way:

- **The encoder looks up to 8 s ahead.** An embedding depends on the whole 8 s block around it.
  Waiting for each block to complete would mean up to 8 s of latency. Instead, every step
  re-encodes the block in progress; only completed blocks are final.
- **The transcript follows the audio in the decoder prompt.** When the embeddings of the block
  in progress change, so does the decoder state for everything after them, transcript included.
  Every step re-feeds the block in progress and the transcript so far.

Completed blocks are encoded once, and the decoder state up to their end is kept. A step
then costs at most one block of encoder work, plus a decoder pass over that block and over the
transcript. As in the reference `qwen-asr` streaming code, the last 5 tokens of the previous
transcript are dropped and decoded again, so the end of a partial transcript can still change.

Consequences:

- Partial transcripts are provisional, and the final one may differ slightly from `batch`
  (punctuation mostly).
- The decoder pass over the transcript grows with it. A long session needs to be cut into
  segments.
- The features use a running maximum for Whisper's floor, which differs from the offline
  features only for bins more than 80 dB below a later peak.

On an RTX 4060 in f32, a 2 s step takes about 0.2 s, and streaming a 44 s file costs about twice
a single batch pass. Re-running the whole transcription on everything heard so far at each step,
as the reference implementation does, costs twelve times as much. Most of a step is spent decoding
tokens one at a time.

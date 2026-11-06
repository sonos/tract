# Kyutai Mimi, streamed

[Mimi](https://huggingface.co/kyutai/mimi) is the neural audio codec behind
Kyutai's Moshi: 24 kHz audio in, 8 codebooks of 2048 entries at 12.5 Hz out,
and back. This example pulses both halves to one 80 ms frame:

- `mimi-encoder.nnef.tgz`: 1920 samples -> 8 codes per pulse
- `mimi-decoder.nnef.tgz`: 8 codes -> 1920 samples per pulse

Both stream with no added delay, and match the batch PyTorch model: codes
exactly, audio to about 1e-6.

```sh
./ci.sh                                   # export, then stream on CPU
cargo run --release --features cuda -- --runtime cuda
cargo run --release --features metal -- --runtime metal
```

`export.py` rewrites Mimi's forward pass in plain torch over the Hugging Face
weights, so it traces cleanly. Convs pad causally on the left, and attention
gets its 250-frame sliding window from a mask built over positions, which
tract's pulsifier turns into a bounded KV window. The program prints
per-frame timings against the 80 ms budget and writes the streamed
reconstruction to `assets/recon-tract.wav`.

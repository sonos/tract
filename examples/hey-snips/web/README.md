# Hey Snips in the browser

Wake word detection on the visitor's microphone, on the CPU through WebAssembly and tract's pulsed
(streaming) graphs. Nothing leaves the machine. Deployed at <https://demo.tract.rs/hey-snips/>.

## How it works

The model is the "Hey Snips" detector of [Coucke et al., 2018](https://arxiv.org/abs/1811.07684):
gated dilated convolutions over 20 log-Mel energies per 10 ms, scoring each 1.8 s window. Model
© Snips and © Sonos.

The page runs it as one NNEF graph that tract streams 80 ms at a time, from the microphone's raw
samples to a score per 10 ms frame:

- a low-pass FIR as a convolution of stride 3, from Web Audio's 48 kHz floats to 16 kHz in the
  int16 range the model was trained on;
- the featurizer: `tract_core_stft` with 25 ms Hamming windows every 10 ms, the power spectrum,
  20 HTK mel bands over 0–8 kHz, and a natural log floored at 1;
- the wavenet.

`export.sh` builds it: tract dumps the TensorFlow model as NNEF, `export.py` puts the resampler
and featurizer in front of it, and tract pulses the whole graph (`--pulse 3840`). The page and
`hey-snips-native` only load that pulsed graph and feed it samples, through the public `tract`
crate with no default features.

The page captures the microphone with an AudioContext at 48 kHz, with echo cancellation, noise
suppression and gain control off, and an audio worklet hands 20 ms blocks to a worker running the
detector. A score above the threshold flags a detection, at most one a second.

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./build.sh
```

`build.sh` fetches the graph from tract's test assets (`GRAPHS_DIR=<dir>` takes a local one, as
`./export.sh <dir>` writes it). `dist/` is then the whole site, to serve as static files over
HTTPS (or on localhost) for microphone access.

## Checking natively

```sh
cargo run --release -p hey-snips-web --bin hey-snips-native -- <hey_snips.nnef.tar> <clip.wav>...
```

streams 48 kHz 16-bit mono clips through the same detector, in the page's 20 ms blocks, and prints
each clip's peak score.

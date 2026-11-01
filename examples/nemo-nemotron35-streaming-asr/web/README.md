# Nemotron speech recognition in the browser

Streaming transcription of the visitor's microphone with NVIDIA's
[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(OpenMDW-1.1), the encoder on the visitor's GPU through WebGPU and tract's `wgpu` runtime.
Nothing leaves the machine.

## How it works

The page downloads the model's `model.safetensors` (2.6 GB, f32) straight from Hugging Face, at a
pinned revision, into the browser's origin private file system, and keeps it there for later
visits; the vocabulary comes from the same revision's `tokenizer.json`. The site serves the four
NNEF graphs of the model (preprocessor, encoder, RNN-T decoder and joint), about 50 MB, whose
weights are `tract_core_named_weight` references to tensors of that file. What the file does not
hold as such stays in the graphs: the positional encodings, the language prompt projector, the
LSTM's summed biases and the mel filter bank.

The worker streams the audio through the model as the native example
[`../src/main.rs`](../src/main.rs) does, and through the same transforms, applied when it loads:

- the microphone, at 48 kHz, decimated to 16 kHz by a windowed-sinc low-pass;
- the preprocessor, pulsed by 100 ms, on the CPU;
- the encoder, with its language prompt fused in, pulsed by its 320 ms attention chunk, on WebGPU;
- greedy RNN-T decoding through the prednet and the joint, on the CPU.

The language is a `lang_id` input of the encoder, or 101 for detection by the model; the model
emits language tags (`<en-US>`), which the page shows as labels.

Everything goes through the public `tract` crate, with these features:

- `pulse`, for pulsifying the preprocessor and the encoder;
- `wgpu`, and `unstable-wgpu-async` for `tract::wgpu::init().await` and `to_host_async().await`:
  a browser cannot block on the GPU;
- `unstable-named-weights` for `Model::link_safetensors`, over a `ReadAt` reading the cached file;
- `transformers` for the attention detection.

A browser build also enables getrandom's `wasm_js` feature (see `Cargo.toml`).

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./build.sh
```

`dist/` is then the whole site, to serve as static files over HTTPS (or on localhost) for
microphone access. `build.sh` takes the graphs from `GRAPHS_URL` (the published tars on R2), or
from a local directory with `GRAPHS_DIR=<dir>`, and writes `build.json`, the commit and date the
page shows next to its source link.

## Making the graphs

```sh
./export.sh <t2n model dir> <model.safetensors> <out dir>
```

from torch_to_nnef's export of the model, as [`../ci.sh`](../ci.sh) makes it (`assets/model`), and
the `model.safetensors` of the revision `site/manifest.json` pins. `export.py` finds each NNEF
tensor's bytes in the safetensors file: a whole tensor becomes a named weight, a run of rows of
one (the LSTM's per-gate matrices) a slice of a named weight, and the rest stays in the tar.

## Checking natively

```sh
cargo run --release -p nemotron-web --bin nemotron-native -- <graphs dir> <hf snapshot dir> <clip.wav> [lang_id]
```

runs the same pipeline on a native WebGPU device (Vulkan, Metal or DX12) and prints the
transcript; the snapshot dir holds `model.safetensors` and `tokenizer.json`. The clip is fed in the
page's 20 ms blocks, at 16 kHz or a multiple of it.

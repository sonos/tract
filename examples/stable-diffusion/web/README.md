# Stable Diffusion 1.5 in the browser

Text to image on the visitor's GPU, through WebGPU and tract's `wgpu` runtime. Nothing leaves the
machine. Deployed at <https://demo.tract.rs/sd15/>.

## How it works

The page downloads Stable Diffusion 1.5's fp16 weights (2.1 GB) straight from Hugging Face, at a
pinned revision, into the browser's origin private file system, and keeps them there for later
visits. The site itself only serves three small NNEF graphs (about 1 MB) whose weights are
`tract_core_named_weight` references to tensors of those safetensors files: tract reads each
tensor at its offset in the cached file when the runtime first needs it.

The model runs in a worker. Each step is one UNet call with classifier-free guidance (batch 2),
stepped by DPM++ 2M Karras (Euler too); the CLIP tokenizer is reimplemented in Rust.

Everything goes through the public `tract` crate, with these features:

- `wgpu`, and `unstable-wgpu-async` for `tract::wgpu::init().await` and `to_host_async().await`:
  a browser cannot block on the GPU;
- `unstable-named-weights` for `Model::link_safetensors`, over a `ReadAt` reading the cached file;
- `transformers` for the attention detection the flash kernel takes.

A browser build also enables getrandom's `wasm_js` feature (see `Cargo.toml`).

Browsers need WebGPU with the `shader-f16` feature: recent Chrome, Edge and Firefox on most GPUs.

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./build.sh
```

`dist/` is then the whole site, to serve as static files. `build.sh` takes the graphs from
`GRAPHS_URL` (the published tars on R2), or from a local directory with `GRAPHS_DIR=<dir>`, and
writes `build.json`, the commit and date the page shows next to its source link.

## Making the graphs

```sh
./export.sh <hf snapshot dir> <out dir>
```

from a snapshot of `stable-diffusion-v1-5/stable-diffusion-v1-5` at the revision
`site/manifest.json` pins. `export.py` writes weight-free fp16 ONNX graphs whose initializers are
external references into the snapshot's safetensors files; tract dumps them to NNEF tars with
`--onnx-lazy-external-data`. Last made with torch 2.11, onnx 1.21, onnxscript 0.6.2,
diffusers 0.37.1 and transformers 5.4.

## Checking natively

```sh
cargo run --release -p sd15-web --bin sd15-native -- <graphs dir> <hf snapshot dir> "a prompt" out.png
```

runs the same pipeline on a native WebGPU device (Vulkan, Metal or DX12); see `src/bin/native.rs`
for its settings.

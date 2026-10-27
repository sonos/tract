# Background replacement in the browser

Person segmentation on the visitor's camera, through WebGPU and tract's `wgpu` runtime, with the
background blurred or replaced by a colour. Nothing leaves the machine.

## How it works

The page offers two segmenters, fetched as ONNX files straight from Hugging Face at a pinned
revision and kept in the browser's cache:

- [MediaPipe Selfie Segmentation](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation)
  (0.5 MB, 256 × 256 mask), fast;
- [MODNet](https://huggingface.co/Xenova/modnet) (26 MB, 512 × 512 alpha matte), finer edges.

Both are under the Apache 2.0 license. `site/models.mjs` lists them with their revisions, sizes
and input normalization.

A worker draws each camera frame into a canvas at the model's resolution and hands it to tract
with `Tensor::from_canvas_normalized`, which copies it onto the GPU and normalizes it there. The
mask goes from the model's output into an RGBA8 texture with `WgpuMask`, and a WebGPU compositor
(`site/compositor.js`) on the same `GPUDevice` blends the frame over the blurred or coloured
background at the camera's resolution, resampling the mask along the image's edges. Frames and
masks never come back to the CPU. The site serves only the page and the wasm.

Only one frame is in flight; frames that arrive while it runs are dropped rather than queued. Every
15th frame also waits for GPU completion between stages to show inference and compositing times:
wall-clock times including queue overhead, not GPU timestamps. Where the browser's completion
callback has a coarse floor (about 100 ms in Firefox) the page shows the frame rate alone.

Everything goes through the public `tract` crate, with the `onnx`, `wgpu` and
`unstable-wgpu-async` features. A browser build also enables getrandom's `wasm_js` feature (see
`Cargo.toml`).

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./build.sh
```

`dist/` is then the whole site, to serve as static files over HTTPS (or on localhost) for camera
access. `build.sh` writes `build.json`, the commit and date the page shows next to its source link.

## Checking natively

```sh
cargo run --release -p pixel-lab-web --bin pixel-lab-native -- <model.onnx> <width> <height> [mean] [std]
```

runs a downloaded model on a native WebGPU device, checks that every compute node is on the GPU
and that the mask matches the CPU's, and times a run. MODNet takes `0.5,0.5,0.5 0.5,0.5,0.5`.

```sh
node --test site/*.test.mjs
```

checks the model list and the timing statistics.

# Background replacement in the browser

Person segmentation on the visitor's camera, as a videoconference would use it, through WebGPU and
tract's `wgpu` runtime, with the background blurred or replaced by a colour. Nothing leaves the
machine.

## How it works

The page offers two segmenters, each in more than one form, fetched as ONNX files straight from
Hugging Face at a pinned revision and kept in the browser's cache:

- MediaPipe Selfie Segmentation (0.5 MB), fast, as its
  [square](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation) (256 × 256 mask)
  and [landscape](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation_landscape)
  (256 × 144 mask) models;
- [MODNet](https://huggingface.co/Xenova/modnet) (26 MB, alpha matte), finer edges, at 512 × 288,
  448 × 256 or 384 × 224: MODNet takes any size divisible by 32, and keeping the camera's 16:9
  shape holds the matte together on close-ups, where a square input that stretches the face breaks
  it up.

All are under the Apache 2.0 license. `site/models.mjs` lists them with their revisions, sizes
and input normalization.

A worker hands camera frames to tract with `Tensor::from_canvas_normalized`, which copies a frame
onto the GPU at the model's size and normalizes it there; the model writes its mask into an RGBA8
texture with `WgpuMask`. A WebGPU compositor (`site/compositor.js`) blends every camera frame over
the blurred or coloured background at the camera's resolution, resampling the mask along the
image's edges, or shows the mask itself.

The compositor runs on a GPU device of its own. A device's queue runs in order, so on tract's
device each composite would wait behind the inference queued before it, and the video would move
at the model's pace. Instead every camera frame is composited at once with the latest mask, and a
new inference starts as soon as the previous mask is ready: a slower model makes the mask lag,
not the video. Each mask crosses from tract's device to the compositor's through one small
readback (256 KB to 600 KB); camera frames stay on the GPU.

The tiles show the camera's frame rate and the composited one, counted when the GPU has finished
a frame; at most three composited frames wait on the GPU, after which camera frames are dropped
rather than queued. Under them, the mask rate and the time from handing a frame to the model to
its mask being ready to composite.

Everything goes through the public `tract` crate, with the `onnx`, `profile` (for the GPU coverage
audit), `wgpu` and `unstable-wgpu-async` features. A browser build also enables getrandom's `wasm_js` feature (see
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

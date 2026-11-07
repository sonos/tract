# Background replacement in the browser

Person segmentation on the visitor's camera, as a videoconference would use it, through WebGPU and
tract's `wgpu` runtime, with the background blurred or replaced by a colour. The camera feed never
leaves the machine.

## How it works

The page offers three segmenters, fetched as ONNX files and kept in the browser's cache:

- MediaPipe Selfie Segmentation (0.5 MB), fast, as its
  [square](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation) (256 × 256 mask)
  and [landscape](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation_landscape)
  (256 × 144 mask) models. The page lists the one whose shape matches the capture size.
- [MODNet](https://huggingface.co/Xenova/modnet) (26 MB, alpha matte), finer edges, at 512 × 288,
  448 × 256 or 384 × 224: one set of weights, three input sizes. MODNet takes any size divisible
  by 32, and keeping the camera's 16:9 shape holds the matte together on close-ups, where a square
  input that stretches the face breaks it up;
- [RVM MobileNetV3](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0)
  (GPL-3.0), recurrent video matting, the official v1.0.0 fp32 ONNX (15 MB).
  The release host does not send the cross-origin header a browser needs, so the page fetches a
  pinned Hugging Face copy of those same bytes. It reads the camera frame at the capture size.
  The downsample ratio defaults to `min(512 / long side, 1)`, so the main network's long side stays at 512
  and a smaller camera is left at scale 1. The page shows that Auto factor as `(R=1.0)` or `(R=0.4)` and can pin
  0.25, 0.5, 0.75, or 1. Pass the factor to `Segmenter.create_rvm` at init, and to `set_ratio` while the
  camera is running. The first use of a factor builds that network; switching back reuses it. Recurrent
  memory is cleared on a change, because its spatial size follows the factor. The mask stays the camera size.

MediaPipe and MODNet are Apache 2.0. `site/models.mjs` lists every entry with its source, size
and input normalization.

A worker hands camera frames to tract with `Tensor::from_canvas_normalized`, which copies a frame
onto the GPU at the model's size and normalizes it there; the model writes its mask into an RGBA8
texture with `WgpuMask`. A WebGPU compositor (`site/compositor.js`) blends every camera frame over
the blurred or coloured background at the camera's resolution, resampling the mask along the
image's edges, or shows the mask itself. The page opens on RVM. Blur strength is Low (5 px) or
High (12 px), High by default, shown only while the background is Blur, and it can change while
the camera is running.

The compositor runs on a GPU device of its own. A device's queue runs in order, so on tract's
device each composite would wait behind the inference queued before it. Drawing stays on the
compositor's device, and a new inference starts as soon as the previous mask is ready. With motion
off, the picture is the camera frame that mask was measured on. That frame is kept until the next
mask replaces it, so a slow model steps the output at the mask rate instead of leaving the cutout
behind a newer frame. With motion on, each new camera frame is shown and the mask is warped onto
it. A repeated call for the same camera frame is ignored, so the output stays
on the camera clock instead of the display refresh. Each mask crosses from tract's device to the
compositor's through one readback.
MediaPipe and MODNet masks are 256 KB to 600 KB. RVM's alpha is the camera frame, so that copy
grows with the capture size. Camera frames stay on the GPU.

Two mask controls sit under the view picker. “Smooth mask edges” is on by default and stands alone:
it softens the mask and pulls the edge onto the camera image whether or not the mask moves.
“Move the mask with the camera” is off unless the visitor turns it on. While it is on, the
compositor keeps a luma pyramid of the frame each inference starts from and, on later frames,
estimates one similarity for the person from textured points, entirely on its own GPU. Bare skin
inherits that motion. A coarse grid may differ from it only where the texture agrees, and smoothness
keeps a hand attached to the arm. The mask is then
read at `x + flow(x)`, the whole step or none of it. Only the outline is compared with the frame the
mask was copied from, and coverage that fails there is removed outright, so a flat background
colour cannot show through as a fringe or as the outline of a hand that has closed. A one-pixel
hole inside that mask is closed. Coverage the mask never had is left empty. With the control off,
the composite shows the camera frame the mask was measured on, so the outline and the person share
that frame's id. The preview keeps running live. With the control on, the newest frame is shown and
the mask is warped from its own frame; if that warp cannot be built, the matched pair is shown instead.

The camera tile shows the camera rate. The output tile shows how often the composited picture
changes: the mask rate while the matched frame is held, and the camera rate while the mask is
moved. Redrawing the same held frame does not count. At most three composited frames wait on the
GPU, after which camera frames are dropped rather than queued. Under the tiles, the mask rate and
the time from handing a frame to the model to its mask being ready to composite.
Below that, the frame budget is inference plus compositing against one frame at
the captured rate. Ingest, inference, compositing and output each keep a short graph
of the samples behind their number.

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
cargo run --release -p pixel-lab-web --bin pixel-lab-native -- --rvm <rvm.onnx> <width> <height> <ratio>
```

runs a downloaded model on a native WebGPU device, checks that every compute node is on the GPU
and that the mask matches the CPU's, and times a run. MODNet takes `0.5,0.5,0.5 0.5,0.5,0.5`.
The `--rvm` form fixes `downsample_ratio` to `ratio` and carries RVM's recurrent state for a second frame.

```sh
node --test site/*.test.mjs
```

checks the model list and the timing statistics.

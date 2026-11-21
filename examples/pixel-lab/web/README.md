# Background replacement in the browser

Person segmentation on the visitor's camera, as a videoconference would use it, through WebGPU and
tract's `wgpu` runtime, with the background blurred or replaced by a colour. The camera feed never
leaves the machine.

## How it works

The page offers MediaPipe, PP-HumanSegV2, webcam MODNet, five Longpipe landscape mattes, and RVM
in fp16 and fp32. MediaPipe and the fp32 RVM are fetched from Hugging Face. The others are written
into `dist/models` when the site is built, then kept in the browser's cache:

- MediaPipe Selfie Segmentation (0.5 MB), fast, as its
  [square](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation) (256 × 256 mask)
  and [landscape](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation_landscape)
  (256 × 144 mask) models. The page lists the one whose shape matches the capture size.
- [PP-HumanSegV2](https://github.com/PaddlePaddle/PaddleSeg) Lite (3.7 MB), the 256 × 144 person
  channel, shown for a 16:9 capture. That sliced ONNX is not published upstream, so
  `prepare-models.py` downloads it from tract-test-assets while the site is built.
- [MODNet webcam](https://github.com/yakhyo/modnet/releases/download/weights/modnet_webcam.onnx)
  (26 MB, alpha matte) at 512 × 288. `prepare-models.py` downloads it while the site is built.
- [Longpipe](https://huggingface.co/Longpipe/longpipe) landscape mattes (MIT), xs 384 × 240,
  small 384 × 224, medium 512 × 320, large 640 × 400, and xl 1280 × 768. The camera and the mask
  stay float32. The body runs in float16, and float16 accumulation is enabled only while that
  model is prepared. `prepare-models.py` downloads the five files while the site is built.
- [RVM MobileNetV3](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0) fp16
  (GPL-3.0, 7.5 MB). The page opens on this model. It is the official v1.0.0 fp16 graph with two
  Casts, so the camera and the matte are float32 and the recurrent state stays float16.
  `prepare-models.py` writes it from the pinned Hugging Face copy of the official file. The loader
  has to accept that mix: a float32 camera, a float32 matte, and float16 recurrent state.
- The same RVM in fp32 (15 MB, GPL-3.0). The release host does not send the cross-origin header a
  browser needs, so the page fetches a pinned Hugging Face copy of those bytes.

Both RVM files read the camera frame at the capture size. The downsample ratio is
`min(512 / long side, 1)`, so the main network's long side stays at 512 and a smaller camera is left
at scale 1. The page shows that factor for the capture size, as `R=0.4` or `R=1.0`, and does not
offer another. The mask stays the camera size.

MediaPipe, PP-HumanSegV2, and MODNet are Apache 2.0. Longpipe is MIT. `site/models.mjs` lists every
entry with its source, size and input normalization.

`?deskmem=640x360` plays `vcd-seated.mp4` from the page directory instead of opening the camera.
The recording is not in the repository. Stop clears that source, so the next Start can load it
again. `&model=` selects a listed model for that capture.

A worker hands camera frames to tract with `Tensor::from_canvas_normalized`, which copies a frame
onto the GPU at the model's size and normalizes it there; the model writes its mask into an RGBA8
texture with `WgpuMask`. A WebGPU compositor (`site/compositor.js`) blends every camera frame over
the blurred, coloured, or still-image background at the camera's resolution, resampling the mask along the
image's edges, or shows the mask itself. Four still images come from
[Livestorm's virtual background library](https://livestorm.co/virtual-background-library), which
offers them free for commercial use: a library, a conference room, a beach, and a mountain path.
A fifth is the white Sonos Five pair from [Sonos.com](https://www.sonos.com/). The page names the image and links its source while it is selected. The page opens on RVM fp16. Blur strength sits on the background row and is shown only while the background is Blur. Low is a
10 px blur after the frame is shrunk by 4, about 20 camera pixels. High is a 12 px blur after the frame is shrunk by 8. High is the default,
and it can change while the camera is running. Harmonize colours
is off by default and appears for a solid colour or a still image. On, it shifts the person's colour
toward that background and leaves the brightness alone.

The compositor runs on a GPU device of its own. A device's queue runs in order, so on tract's
device each composite would wait behind the inference queued before it. Drawing stays on the
compositor's device, and a new inference starts as soon as the previous mask is ready. With motion
off, the picture is the camera frame that mask was measured on. That frame is kept until the next
mask replaces it, so a slow model steps the output at the mask rate instead of leaving the cutout
behind a newer frame. With motion on, the frame inference is computing waits for its own mask.
A later camera frame, which will not get a mask of its own, is shown with the last mask moved onto it.
A repeated call for the same camera frame is ignored, so the output stays
on the camera clock instead of the display refresh. Each mask crosses from tract's device to the
compositor's through one readback. On Firefox, while that readback or a composite is waiting to
finish, the page submits an empty command so the GPU process polls instead of waiting on its 100 ms
timer. Chrome and Safari are unchanged.
MediaPipe and MODNet masks are 256 KB to 600 KB. RVM's alpha is the camera frame, so that copy
grows with the capture size. Camera frames stay on the GPU.

The mask controls sit under the view picker. “Smooth mask edges” is on by default and stands alone:
it softens the mask and pulls the edge onto the camera image whether or not the mask moves.
“Hold uncertain mask” is on by default for the single-frame models, and hidden for RVM. Once per new
mask, at the mask's resolution, each new score is mixed toward the previous blended score by how
close it sits to one half. A sure 0 or 1 replaces that history. A score near one half keeps most of
it. The picture then shows background at or below 0.1 and solid person at or above 0.7. The next
mask blends against the mix, before that picture step. “Smooth mask edges” is the separate step
that feathers the outline onto the camera picture. It does not remember the previous frame. The first mask passes through. Turning the hold off
forgets the previous mask. RVM already carries a recurrent state, so the blend is not applied to it.
“Move the mask with the camera” is off unless the visitor turns it on. While it is off, the
composite shows the camera frame the mask was measured on, so the output steps at the mask rate.
While it is on, a frame that inference is computing waits for that mask and is shown as that pair.
A camera frame that arrives during that wait will not get a mask of its own, so the last mask is
moved onto it and the output stays on the camera clock. The compositor keeps a luma pyramid of the
frame each inference starts from and estimates one similarity for the person from textured points,
entirely on its own GPU. Bare skin inherits that motion. A coarse grid may differ from it only where
the texture agrees, and smoothness keeps a hand attached to the arm. The mask is then read at
`x + flow(x)`, the whole step or none of it. Only the outline is compared with the frame the mask
was copied from, and coverage that fails there is removed outright, so a flat background colour
cannot show through as a fringe or as the outline of a hand that has closed. A one-pixel hole inside
that mask is closed. Coverage the mask never had is left empty. If the warp cannot be built, the
matched pair is shown instead. The preview keeps running live.

The camera tile shows the camera rate. The output tile shows how often the composited picture
changes: the mask rate while the matched frame is held, and the camera rate while the mask is
being moved forward to fill the gap. Redrawing the same held frame does not count, and neither does the true mask
replacing the warp on the camera frame already on screen. At most three composited frames wait on the
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
It also runs `prepare-models.py`, which needs network access and the `onnx` Python package, to
download the webcam MODNet and the PP-HumanSegV2 slice and write the fp16 RVM into `dist/models`.
When `python3` cannot import `onnx`, `build.sh` installs a pinned one into a venv under `target/`
and reuses it on later builds.

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

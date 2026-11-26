# Bezier Piano

A webcam instrument. The camera fills the stage, cubic Bezier strings lie over it, and a fingertip
that crosses a string plays a note. Pitch comes from which string you crossed and where along it;
the fingertip's speed sets the loudness.

tract runs Google's MediaPipe hand models in WebAssembly, in a worker:

- the palm detector (192 × 192, 2016 SSD anchors) finds each hand in a letterboxed frame;
- the landmark model (224 × 224) reads a square crop around the palm, rotated so the fingers point
  up, and places 21 points;
- while a hand stays in view, its landmarks give the next frame's crop and the detector only runs
  to look for a second hand.

Both are TFLite files from the float16
[Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) bundle
(© Google, Apache-2.0). The page downloads it from Google's model bucket on first load and checks
its SHA-256; tract reads the two `.tflite` entries out of the bundle as they are. Anchors,
non-maximum suppression, the crops and the One Euro smoothing are in `site/tracking.mjs`.

**Backend** picks where tract runs them: its wgpu runtime on WebGPU (the default), or its CPU
kernels in wasm with simd128. The choice sticks, and `?backend=cpu` or `?backend=wgpu` in the URL
overrides it. A browser without WebGPU in workers starts on the CPU. Both backends are built into one
wasm; the models are prepared for a backend the first time it is picked.

The piano voice is additive WebAudio: six slightly inharmonic partials, a fast attack, an
exponential decay and a few milliseconds of hammer noise.

## Build and run

```sh
examples/bezier-piano/web/build.sh
python3 -m http.server -d examples/bezier-piano/web/dist 8000
```

Then open <http://127.0.0.1:8000/> and press **Start camera**: the camera needs localhost or HTTPS.
`build.sh` needs the `wasm32-unknown-unknown` target and the wasm-bindgen CLI version `Cargo.lock`
pins.

`node --test site/*.test.mjs` checks the string crossings, the scale mapping and the curve fit, and
the tracking geometry. `cargo run --release -p bezier-piano-web --bin bezier-piano-native --
hand_landmarker.task` times the two models natively (`--wgpu` before the file for the wgpu runtime); with `detect` or `landmarks` and two raw f32
files it runs one model, for comparing with another runtime.

## What you can play

- **Strings.** Two to twelve. Six strings are C3 to C8, top string highest.
- **Scale.** Major or minor pentatonic, major, natural minor. Position along a string quantizes to
  a degree of that scale.
- **Mirror** flips the camera and the strings together. **Show landmarks** draws the 21 points.
- **Draw a string.** Pinch thumb to index and drag; releasing fits the stroke to a cubic. Without
  draw mode, a pinch grabs the nearest control point.

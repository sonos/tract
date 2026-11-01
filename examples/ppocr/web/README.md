# Read a picture in the browser

PP-OCRv6 finds the lines in a picture and reads them, through tract compiled to
WebAssembly. The page can run the same graphs on the CPU or on WebGPU. A sample
page is already loaded. A visitor can upload another picture or take a camera
snapshot. Nothing leaves the machine except the model download.

## How it works

The selector offers the three PP-OCRv6 tiers published by PaddlePaddle. Each one
is fetched from Hugging Face at a pinned revision and kept in the browser cache.
The character dictionary travels in that tier's `inference.yml`. Tiny uses the
short dictionary. Small and Medium share the long one. Older PP-OCR lines are
different graphs and are not in the selector.

| Tier | Detection | Recognition | Download |
| --- | --- | --- | --- |
| Tiny | `PaddlePaddle/PP-OCRv6_tiny_det_onnx` @ `2ba1506c` | `PP-OCRv6_tiny_rec_onnx` @ `2612ab37` | 6 MB |
| Small | `PP-OCRv6_small_det_onnx` @ `28fe5895` | `PP-OCRv6_small_rec_onnx` @ `b8f84f0b` | 30 MB |
| Medium | `PP-OCRv6_medium_det_onnx` @ `61323801` | `PP-OCRv6_medium_rec_onnx` @ `50c7eaca` | 132 MB |

`site/models.mjs` has the full revisions and byte sizes. The weights are
Apache-2.0.

Preprocessing, the text boxes and CTC stay in `site/ocr.mjs`. tract runs the
two ONNX graphs. A concrete input size is prepared the first time it is seen:
detection varies with the picture, and each text line has its own width. The
GPU switch builds a second set of plans on the `wgpu` runtime. In a browser
that runtime's outputs stay on the device until `Tensor::to_host_async`.

The crate uses the public `tract` facade with `onnx`, `wgpu` and
`unstable-wgpu-async`, and no default features.

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./build.sh
```

`dist/` is the static site. Serve it over HTTPS, or on localhost, so the camera
can be used. `build.sh` writes `build.json`, the commit and date the page shows
next to its source link.

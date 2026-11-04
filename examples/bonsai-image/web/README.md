# Bonsai Image in the browser

Text to image on the visitor's GPU, through WebGPU and tract's `wgpu` runtime. Nothing leaves the
machine. Browsers need WebGPU with the `shader-f16` feature.

## How it works

The page downloads the model once into the browser's origin private file system. A worker reads
each stage from there, keeps it loaded for the next image, and shows the transformer's estimate of
the image after each step: `x - sigma * v` through a small linear map to RGB, fit by
`fit_preview.py`.

Every size shares one weights tar per stage. `split_stages.py` keeps what differs between sizes in
a small `<stage>-<HxW>.tar`, which reads as one stream with `<stage>.weights.tar`.

The encoder and the transformer both take 64 text tokens, so a prompt has 52 once the chat
template is in.

## Building the site

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the version Cargo.lock pins>
./export.sh <hf snapshot dir> <models dir>
MODELS_DIR=<models dir> ./build.sh
```

`dist/` is then the whole site, to serve as static files. `MODELS_URL` points the manifest at a
published copy of the models instead of linking them in.

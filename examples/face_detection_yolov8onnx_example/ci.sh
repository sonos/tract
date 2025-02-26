#!/bin/bash

set -ex
export RUSTUP_TOOLCHAIN=1.75.0

# download pre-exported onnx model
wget -Nq "https://tract-ci-builds.s3.amazonaws.com/model/yolov8n-face.onnx"

# on win/linux 
cargo run -- --input-image grace_hopper.jpg --weights yolov8n-face.onnx 

wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasi
cargo build --target wasm32-wasi
wasmtime --dir . ../../target/wasm32-wasi/debug/face_detection_yolov8onnx_example.wasm --input-image grace_hopper.jpg --weights yolov8n-face.onnx

cargo clean 
rm yolov8n-face.onnx

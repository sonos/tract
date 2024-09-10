#!/bin/bash

set -ex

# download pre-exported onnx model
wget -q "https://tract-ci-builds.s3.amazonaws.com/model/yolov8n-face.onnx"

# on win/linux 
cargo run -- --input-image grace_hopper.jpg --weights yolov8n-face.onnx 

wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasi
cargo build --target wasm32-wasi --release
wasmtime --dir . ../../target/wasm32-wasi/release/face_detection_yolov8onnx_example.wasm --input-image grace_hopper.jpg --weights yolov8n-face.onnx

cargo clean 
rm yolov8n-face.onnx

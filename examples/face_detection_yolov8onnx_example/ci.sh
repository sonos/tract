#!/bin/bash

set -ex
# download pre-exported onnx model
wget -Nq "https://tract-test-assets.tract.rs/yolov8n-face.onnx"

# on win/linux 
cargo run -- --input-image grace_hopper.jpg --weights yolov8n-face.onnx 

wasmtime -V || curl --retry 5 --retry-delay 5 -sSf https://wasmtime.dev/install.sh | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasip1
cargo build --target wasm32-wasip1 --release
wasmtime --dir . ../../target/wasm32-wasip1/release/face_detection_yolov8onnx_example.wasm --input-image grace_hopper.jpg --weights yolov8n-face.onnx

rm yolov8n-face.onnx

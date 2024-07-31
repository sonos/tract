#!/bin/bash

set -ex

# download pre-exported onnx model
wget -O yolov8n-face.onnx --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1PYAG1ypAuwh_rDROaUF0OdLmBqOefBGL"

# on win/linux 
cargo run -- --input-image grace_hopper.jpg --weights yolov8n-face.onnx 

wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasi
cargo build --target wasm32-wasi
wasmtime --dir . ../../target/wasm32-wasi/debug/face_detection_yolov8onnx_example.wasm --input-image grace_hopper.jpg --weights yolov8n-face.onnx

cargo clean 
rm yolov8n-face.onnx

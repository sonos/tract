#!/bin/bash

set -ex

# download pre-exported onnx model
wget -O yolov8n-face.onnx --no-check-certificate -r "https://drive.google.com/uc?export=download&id=1PYAG1ypAuwh_rDROaUF0OdLmBqOefBGL"
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx

# on win/linux 
cargo run -- --face1 grace_hopper.jpeg --face2 grace_hopper2.jpeg 

wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasi
cargo build --target wasm32-wasi
wasmtime --dir . ../../target/wasm32-wasi/debug/face_similarity_arcface_onnx.wasm --face1 grace_hopper.jpeg --face2 grace_hopper2.jpeg

cargo clean 
rm yolov8n-face.onnx
rm arcfaceresnet100-8.onnx

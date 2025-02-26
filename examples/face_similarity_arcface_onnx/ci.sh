#!/bin/bash

set -ex
export RUSTUP_TOOLCHAIN=1.75.0

# download pre-exported onnx model
wget -Nq "https://tract-ci-builds.s3.amazonaws.com/model/yolov8n-face.onnx"
wget -Nq "https://tract-ci-builds.s3.amazonaws.com/model/arcfaceresnet100-8.onnx"

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

#!/bin/bash

set -ex

# download pre-exported onnx model
wget -Nq "https://tract-ci-builds.s3.amazonaws.com/model/yolov8n-face.onnx"
wget -Nq "https://tract-ci-builds.s3.amazonaws.com/model/arcfaceresnet100-8.onnx"

# on win/linux 
cargo run --release -- --face1 grace_hopper.jpeg --face2 grace_hopper2.jpeg 

wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasip1
cargo build --target wasm32-wasip1 --release
wasmtime --dir . ../../target/wasm32-wasip1/release/face_similarity_arcface_onnx.wasm --face1 grace_hopper.jpeg --face2 grace_hopper2.jpeg

rm yolov8n-face.onnx
rm arcfaceresnet100-8.onnx

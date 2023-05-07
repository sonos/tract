#!/bin/sh

set -ex

wget -q https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx -O mobilenetv2-7.onnx

# on win/linux
cargo run
# on wasm
wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
rustup target install wasm32-wasi
cargo build --target wasm32-wasi
wasmtime ../../target/wasm32-wasi/debug/example-onnx-mobilenet-v2.wasm --dir=.

cargo clean
rm  mobilenetv2-7.onnx

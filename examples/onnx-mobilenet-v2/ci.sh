#!/bin/sh

set -ex

[ -e mobilenetv2-7.onnx ] || \
    wget -q https://s3.amazonaws.com/tract-ci-builds/tests/mobilenetv2-7.onnx -O mobilenetv2-7.onnx

# on win/linux
cargo run
# on wasm
wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasip1
cargo build --target wasm32-wasip1
wasmtime --dir . ../../target/wasm32-wasip1/debug/example-onnx-mobilenet-v2.wasm

cargo run --bin dyn-shape

rm  mobilenetv2-7.onnx

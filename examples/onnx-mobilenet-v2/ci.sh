#!/bin/sh

set -ex
export RUSTUP_TOOLCHAIN=1.75.0

[ -e mobilenetv2-7.onnx ] || \
    wget -q https://s3.amazonaws.com/tract-ci-builds/tests/mobilenetv2-7.onnx -O mobilenetv2-7.onnx

# on win/linux
cargo run
# on wasm
wasmtime -V || curl https://wasmtime.dev/install.sh -sSf | bash # install wasmtime
PATH=$PATH:$HOME/.wasmtime/bin
rustup target install wasm32-wasi
cargo build --target wasm32-wasi
wasmtime --dir . ../../target/wasm32-wasi/debug/example-onnx-mobilenet-v2.wasm

cargo clean
rm  mobilenetv2-7.onnx

#!/bin/sh

set -ex

cargo install bindgen-cli
cargo install cbindgen

cbindgen ffi > tract.h
cp tract.h c
bindgen tract.h -o proxy/sys/bindings.rs --rust-target 1.75.0

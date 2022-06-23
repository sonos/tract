#!/bin/sh

set -ex

rustup toolchain add nightly
rustup component add rust-src --toolchain nightly-x86_64-unknown-linux-gnu

export RUSTFLAGS=-Zsanitizer=address 
export RUSTDOCFLAGS=$RUSTFLAGS
export RUSTUP_TOOLCHAIN=nightly
export RUST_VERSION=nightly
export CARGO_EXTRA="-Zbuild-std --target x86_64-unknown-linux-gnu"

./.travis/regular-tests.sh
./.travis/onnx-tests.sh
./.travis/cli-tests.sh


#!/bin/sh

set -ex

TARGET=$(rustc -vV | sed -n 's|host: ||p')

rustup toolchain add nightly
rustup component add rust-src --toolchain nightly-$TARGET

export RUSTFLAGS=-Zsanitizer=address 
export RUSTDOCFLAGS=$RUSTFLAGS
export RUSTUP_TOOLCHAIN=nightly
export RUST_VERSION=nightly
export CARGO_EXTRA="-Zbuild-std --target $TARGET"

./.travis/regular-tests.sh
if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/onnx-tests.sh
if [ -n "$CI" ]
then
    cargo clean
fi
./.travis/cli-tests.sh


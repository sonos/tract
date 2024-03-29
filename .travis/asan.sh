#!/bin/sh

set -ex

TARGET=$(rustc -vV | sed -n 's|host: ||p')

if [ -n "$GITHUB_ACTIONS" ]
then
    pip3 install numpy
fi

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


#!/bin/sh

set -ex

# RUSTFLAGS=-Zsanitizer=address cargo +nightly test -Zbuild-std --target $(rustc -vV | sed -n 's|host: ||p')

TARGET=$(rustc -vV | sed -n 's|host: ||p')

rustup toolchain add nightly
rustup component add rust-src --toolchain nightly-$TARGET

export RUSTFLAGS=-Zsanitizer=address 
export RUSTDOCFLAGS=$RUSTFLAGS
export RUSTUP_TOOLCHAIN=nightly
export RUST_VERSION=nightly
export CARGO_EXTRA="--target $TARGET"

cargo -q test -q -p tract-core --features paranoid_assertions $CARGO_EXTRA

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


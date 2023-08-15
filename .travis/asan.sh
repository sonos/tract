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

df -h
./.travis/regular-tests.sh
df -h
du -hs target
du -hs .cached
./.travis/onnx-tests.sh
df -h
du -hs target
du -hs .cached
# ./.travis/cli-tests.sh
# df -h


#!/bin/sh

set -ex

ARCH=$1

case $ARCH in
    aarch64)
        MUSL_TRIPLE=aarch64-linux-musl
        RUST_TRIPLE=aarch64-unknown-linux-musl
    ;;
    armv7)
        MUSL_TRIPLE=armv7l-linux-musleabihf
        RUST_TRIPLE=armv7-unknown-linux-musleabi
    ;;
    *)
        exit "Can't build with musl for $ARCH"
    ;;
esac


# only works starting with 1.48
export RUSTUP_TOOLCHAIN=1.48.0

rustup target add $RUST_TRIPLE

curl -s https://musl.cc/${MUSL_TRIPLE}-cross.tgz | tar zx

MUSL_BIN=`pwd`/${MUSL_TRIPLE}-cross/bin
export PATH=$MUSL_BIN:$PATH

export TARGET_CC=$MUSL_BIN/${MUSL_TRIPLE}-gcc

RUST_TRIPLE_ENV=$(echo $RUST_TRIPLE | tr 'a-z-' 'A-Z_')
export CARGO_TARGET_${RUST_TRIPLE_ENV}_CC=$TARGET_CC
export CARGO_TARGET_${RUST_TRIPLE_ENV}_LINKER=$TARGET_CC

cargo build --target $RUST_TRIPLE --release -p tract

mv target/${RUST_TRIPLE}/release/tract target/tract

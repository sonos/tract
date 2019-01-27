#!/bin/sh

set -ex

mkdir -p $HOME/cached/bin
ls $HOME/cached/bin
PATH=$HOME/cached/bin:$PATH

which cargo-dinghy || cargo install --debug --root $HOME/cached cargo-dinghy

sudo apt-get install git

case "$PLATFORM" in
    "raspbian")
        [ -e $HOME/cached/raspitools ] || git clone https://github.com/raspberrypi/tools $HOME/cached/raspitools
        TOOLCHAIN=$HOME/cached/raspitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf
        export RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
        rustup target add $RUSTC_TRIPLE
        echo "[platforms.$PLATFORM]\nrustc_triple='$RUSTC_TRIPLE'\ntoolchain='$TOOLCHAIN'" > $HOME/.dinghy.toml
        cargo dinghy --platform $PLATFORM build --release -p tract
        cargo dinghy --platform $PLATFORM bench --no-run -p tract-linalg
    ;;
    "aarch64")
        sudo apt-get -y install binutils-aarch64-linux-gnu gcc-4.8-aarch64-linux-gnu
        export RUSTC_TRIPLE=aarch64-unknown-linux-gnu
        rustup target add $RUSTC_TRIPLE
        export TARGET_CC=aarch64-linux-gnu-gcc-4.8
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc-4.8
        cargo build --target $RUSTC_TRIPLE --release -p tract
        cargo build --target $RUSTC_TRIPLE --release --benches -p tract-linalg
    ;;
    *)
esac

if [ -n "$AWS_ACCESS_KEY_ID" -a "$TRAVIS_EVENT_TYPE" = "push" ]
then
    TASK_NAME=`.travis/make_bundle.sh`
    aws s3 cp $TASK_NAME.tgz s3://tract-ci-builds/tasks/$PLATFORM/$TASK_NAME.tgz
fi

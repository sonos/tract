#!/bin/sh

set -ex

mkdir -p $HOME/cached/bin
PATH=$HOME/cached/bin:$PATH

which cargo-dinghy || cargo install --debug --root $HOME/cached cargo-dinghy

sudo apt install git

case "$PLATFORM" in
    "raspbian")
        [ -e $HOME/cached/raspitools ] || git clone https://github.com/raspberrypi/tools $HOME/cached/raspitools
        rustup target add arm-unknown-linux-gnueabihf
        TOOLCHAIN=$HOME/cached/raspitools/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf
        RUSTC_TRIPLE=arm-unknown-linux-gnueabihf
        export RUSTC_TRIPLE
        echo "[platforms.$PLATFORM]\nrustc_triple='$RUSTC_TRIPLE'\ntoolchain='$TOOLCHAIN'" > $HOME/.dinghy.toml
    ;;
    *)
esac

if [ -n "$AWS_ACCESS_KEY_ID" ]
then
    cargo dinghy --platform $PLATFORM build --release -p tract
    TASK_NAME=`.travis/make_bundle.sh`
    aws s3 cp $TASK_NAME.tgz s3://tract-ci-builds/tasks/$PLATFORM/$TASK_NAME.tgz
else
    cargo dinghy --platform $PLATFORM build -p tract
fi

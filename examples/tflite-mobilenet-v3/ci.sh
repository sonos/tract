#!/bin/sh

set -ex

wget -q "https://tract-ci-builds.s3.amazonaws.com/model/mobilenet_v3_small_100_224.tflite" -O mobilenet_v3_small_100_224.tflite
cargo run
rm -rf mobilenet*
cargo clean

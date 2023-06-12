#!/bin/sh

set -ex

wget -q "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1?lite-format=tflite" -O mobilenet_v3_small_100_224.tflite
cargo run
rm -rf mobilenet*
cargo clean

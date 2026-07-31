#!/bin/sh

set -ex

wget -q "https://tract-test-assets.tract.rs/mobilenet_v3_small_100_224.tflite" -O mobilenet_v3_small_100_224.tflite
cargo run
rm -rf mobilenet*

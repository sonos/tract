#!/bin/sh

set -ex

wget -q https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz -O mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz
cargo run
rm -rf mobilenet*
cargo clean

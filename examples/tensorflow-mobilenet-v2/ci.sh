#!/bin/sh

set -ex

wget -nc -q https://tract-test-assets.tract.rs/mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz
cargo run
rm -rf mobilenet*

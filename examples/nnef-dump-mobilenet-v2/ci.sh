#!/bin/sh

set -ex

[ -e mobilenet_v2_1.4_224.tgz ] || wget -q https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz

cargo run -p tract -- mobilenet_v2_1.4_224_frozen.pb -i 1,224,224,3,f32 dump --nnef mobilenet.nnef.tgz
cargo run

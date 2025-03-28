#!/bin/sh

set -ex

wget -nc -q https://s3.amazonaws.com/tract-ci-builds/model/mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz

cargo run -p tract -- mobilenet_v2_1.4_224_frozen.pb -i 1,224,224,3,f32 dump --nnef mobilenet.nnef.tgz
cargo run

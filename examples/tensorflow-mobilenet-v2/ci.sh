#!/bin/sh

set -ex

wget -nc -q https://s3.amazonaws.com/tract-ci-builds/model/mobilenet_v2_1.4_224.tgz
tar zxf mobilenet_v2_1.4_224.tgz
cargo run
rm -rf mobilenet*
cargo clean

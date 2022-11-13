#!/bin/sh

set -ex

wget -q https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx -O mobilenetv2-7.onnx
cargo run

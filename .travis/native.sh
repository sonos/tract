#!/bin/sh

set -ex

./.travis/regular-tests.sh
./.travis/onnx-tests.sh
./.travis/cli-tests.sh

#!/bin/sh

set -ex

./.travis/regular-tests.sh

# useful as debug_asserts will come into play
cargo -q test -q -p onnx-test-suite -- --skip real_

./.travis/onnx-tests.sh
./.travis/cli-tests.sh

#!/bin/sh

set -ex

rustup update

./.travis/regular-tests.sh

cargo test -p tract-core --features paranoid_assertions

# useful as debug_asserts will come into play
cargo -q test -q -p onnx-test-suite -- --skip real_

./.travis/onnx-tests.sh
./.travis/cli-tests.sh

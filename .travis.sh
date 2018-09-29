#!/bin/sh

set -ex

export CI=true

cargo build --release
cargo test --release
cargo check --benches -all # running benches on travis is useless
cargo doc
(cd tfdeploy-tf ; cargo test --release)
(cd tfdeploy-onnx ; env -u RUST_BACKTRACE cargo test --release)
(cd conform ; cargo test --release)
(cd exs/inceptionv3 ; cargo test --release)
(cd cli ; cargo test --release)

# Tensorflow protobuf code gen

This used to be a build.rs script, depending on protoc present on the system.
It was cumbersome for all kind of reasons, specifically it was breaking docs.rs
generation. So let's put the handful of generated files in git.

```
cargo run && cp target/generated/tensorflow/*.rs ../tensorflow/src/tfpb
```

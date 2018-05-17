# TF deploy / Rust

[![Build Status](https://travis-ci.org/kali/tensorflow-deploy-rust.svg?branch=master)](https://travis-ci.org/kali/tensorflow-deploy-rust)

A tiny TensorFlow inference-only executor.

## Why ?

TensorFlow is a big beast. It is designed for being as efficient as possible
when training NN models on big platforms, with as much support for custom
hardware as practical.

Performing inference, aka running trained models sometimes needs to happen
on small-ish devices, like mobiles phones or single-board computers.
Cross-compiling TensorFlow for these platforms can be a daunting task, and
does produce a huge libraries.

`TFDeploy` is a tensorflow-compatible inference library. It loads a tensorflow 
frozen model from the regular protobuf format, and runs data through it.

## Status

Even if `TFDeploy` is very far from supporting any arbitrary model, it can run
Google Inception v3, or Snips hotword models, and missing operators are easy
to add.

## Roadmap

One important guiding cross-concern: this library must cross-compile as
easily as practical to small-ish devices (think 30$ boards).

* integrate other TF models to use as example, test and bench
    * https://github.com/ARM-software/ML-KWS-for-MCU
    * https://github.com/mozilla/DeepSpeech
* investigate alternative impls for Conv2D, and dilated convolutions
* consider having a separate set of non-TF mimicking operators
    * optimise some ops combination (mul followed by add -> GEMM for instance)
    * support kaldi

# License

Note: files in the `protos/tensorflow` directory are copied from the
[TensorFlow](https://github.com/tensorflow/tensorflow) project and are not
covered by the following licence statement.

## Apache 2.0/MIT

All original work licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
     at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

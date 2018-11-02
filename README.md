# Tract

[![Build Status](https://travis-ci.org/kali/tract.svg?branch=master)](https://travis-ci.org/kali/tract)
[![Doc](https://docs.rs/tract-core/badge.svg)](https://docs.rs/tract-core)

A tiny TensorFlow and ONNX inference engine.

_This project used to be called tfdeploy, or Tensorflow-deploy-rust._

## What ?

`Tract` is a tensorflow-compatible inference library. It loads a Tensorflow
or ONNX frozen model from the regular protobuf format, and flows data through
it.

## Status and compatibility

### ONNX

As of today (nov 2018), Tract passes successfully about 85% of ONNX backends
tests. sqeezenet, densenet121 resnet50, inception_v2, vgg19 tests are passing.

### Tensorflow

Even if `Tract` is very far from supporting any arbitrary model, it can run
Google Inception v3, Snips wakeword models, missing operators are easy
to add. The lack of easy to reuse test suite, and the wide diversity of 
operators in Tensorflow make it difficult to target a full support.

### Streaming capability ("pulse")

This is a semi-experimental support for real-time application like voice
processing. It is similar in purpose to the way kaldi decodes voice: decoding
must happens "as you go", you can not wait for the end of the incoming audio
to start decoding.

Our current approach to the implementation is a bit different though. We
convert a regular network to a pulse network that will act on small time
slices (the "pulses"). This conversion only makes sense for a subset of 
operators, so not all networks can be converted to a pulse network.

## Roadmap

One important guiding cross-concern: this library must cross-compile as
easily as practical to small-ish devices (think 20$ boards).

* complete ONNX support, and propose it as backend
* integrate other TF models to use as example, test and benches
    * https://github.com/ARM-software/ML-KWS-for-MCU
    * https://github.com/mozilla/DeepSpeech
* consider acting as kaldi backend

# License

Note: files in the `protos/tensorflow` directory are copied from the
[TensorFlow](https://github.com/tensorflow/tensorflow) project and are not
covered by the following licence statement.

Note: files in the `protos/onnx` directory are copied from the
[ONNX](https://github.com/onnx/onnx) project and are not
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

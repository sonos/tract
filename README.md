# TF deploy / Rust

[![Build Status](https://travis-ci.org/kali/tensorflow-deploy-rust.svg?branch=master)](https://travis-ci.org/kali/tensorflow-deploy-rust)

A tiny TensorFlow inference-only executor.

## Why ?

TensorFlow is a big beast. It is designed for being as efficient as possible
when training NN models on big platforms, with as much support for custom
hardware as practical.

Performing inference, aka running trained models sometimes needs to happen
on small-ish devices, mobiles phones and stuff. Cross-compiling TensorFlow for
these platforms can be a daunting task, and produces huge libraries.

This project started as a very pragmatic answer to a critical problem we
encountered at Snips recently: we needed to run (tiny) model as part of a
library that we were porting to Android. The inference-only C interface that we
were relying on other platforms (`libtensorflow`) was not provided nor
buildable for Android. We wasted so much time trying that we decided we needed
another option and we started this project on the side as a plan B.

It turns out we finally managed to build `libtensorflow` in time. As a matter
of fact, TensorFlow team released their own Android build scripts just a few
days after we managed to craft ours.

So this project is only a hobby of mine right now.

## Status

This is very far to support any arbitrary model. Right now, we have a skeleton
interpreter, and only a handful of naive implementation for actual Ops. Just
what we needed for Google's Inception v3 to run. Moreover, only the strictly
necessary data types have been implemented (most operators right now only
operate on f32, a handful on integers).

Adding an Op is relatively straightforward, adding a data type more
complicated.

## Roadmap

One important guiding cross-concern: I want this library to cross-compile as
easily as practical to small-ish devices (think 30$ boards).

* cleanup and generalize (op-wise and type-wise) basic operators (arithmetic, shape). consider factorizing paramater reading code and datatype switching 
* find and integrate other TF models to use as example, test and bench
* investigate alternative impls for Conv2D
* refactor interpreter: make it stack-based (because it's easy) and stop cloning everything
* consider ops accepting borrowed matrixes when it makes sense to avoid more clones
* consider having a separate set of non-TF mimicking operators
* optimise some ops combination (mul followed by add -> GEMM for instance)

# License

Note: files in the `protos` directory are copied from the
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

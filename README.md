# TF deploy / Rust

A tiny TensorFlow inference-only executor.

## Status

This is very far to support any arbitrary model. Right now, we have a skeleton
interpreter, with support for f32 only, and only a handful of naive
implementation for actual Ops. Just what we needed for one tiny model, really.

Adding an Op is relatively straightforward, adding a data type more
complicated.

## Why ?

TensorFlow is a big beast. It is designed for being as efficient as possible
when training NN models on big platforms, with as much support for custom
hardware as practical.

Performing inference, that is running trained models, sometimes need to happen
on small-ish devices, mobiles phones and stuff. Cross-compiling TensorFlow for
these platforms can be a daunting task, produce a huge library.

This is a very pragmatic answer to a very acute problem we encountered
recently: we needed to run (tiny) model as part of a library that we need to
port to Android. The inference-only C interface that we were used to on other
platforms can not be build out of the box for Android. We wasted so much time
trying that we decided we needed another option.

Our model was so simple that hard-coding it was actually an option but we 
found out about [tfdeploy](https://github.com/riga/tfdeploy), and realised
we could do the same in our favourite language.

## Why ?

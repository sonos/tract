![tract-logo](assets/tract-logo/PNG/tract-horizontal-blue.png)

![MIT/Apache 2](https://img.shields.io/crates/l/tract)
[![Native tests status](https://github.com/snipsco/tract/workflows/Native%20test%20suite/badge.svg)](https://github.com/snipsco/tract/actions)
[![Linux targets status](https://github.com/snipsco/tract/workflows/Linux%20targets/badge.svg)](https://github.com/snipsco/tract/actions)
[![Doc](https://docs.rs/tract-core/badge.svg)](https://docs.rs/tract-core)

Snips' tiny TensorFlow and ONNX inference engine.

_This project used to be called tfdeploy, or Tensorflow-deploy-rust._

## What ?

`tract` is a tensorflow- and ONNX- compatible inference library. It loads a 
Tensorflow or ONNX frozen model from the regular protobuf format, and flows
data through it.

## Quick start

* [MobileNet v2 with TensorFlow](examples/tensorflow-mobilenet-v2)

## Real-time streaming support

This is a semi-experimental support for real-time applications like voice
processing. In many real time voice applications, processing must happen "as you
go". One can not wait for the end of the incoming audio signal to start
decoding.

While Kaldi has built its inference engine around this streaming constraint,
our approach to the same issue is a bit different. `tract` graph analyser and
optimiser will reason on "streamed" tensors, in order to generate an equivalent
stateful "pulsing" network that will propagate small time slices ("pulses") of
data. This makes optimisation efforts on pulsing and "finite" tensor modes
mutually benefit each other.

Obviously, this conversion only makes sense for a subset of operators, so not
all networks can be converted to a pulse network: for instance, an aggregation
(like a SoftMax) on the time dimension can only be given a value when the
signal has been processed up to the end.

## Status and compatibility

### ONNX

As of today (May 2019), `tract` passes successfully about 85% of ONNX backends
tests. `squeezenet`, `densenet121`, `resnet50`, `inception_v2` and `vgg19` tests
are passing.

Covering all of ONNX and making a ONNX backend out of `tract` is on the roadmap.

The following operators are implemented and tested:

Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool,
BatchNormalization, Cast, Ceil, Clip, Concat, Constant, ConstantLike,
ConstantOfShape, Conv, Cos, Cosh, Div, Dropout, Elu, Equal, Exp, Expand,
EyeLike, Flatten, Floor, Gather, Gemm, GlobalAveragePool, GlobalLpPool,
GlobalMaxPool, Greater, HardSigmoid, Hardmax, Identity, IsNaN, LRN, LSTM,
LeakyRelu, Less, Log, LogSoftmax, MatMul, Max, MaxPool, Mean, Min, Mul, Neg,
Not, Or, PRelu, Pad, ParametricSoftplus, Pow, Reciprocal, ReduceL1, ReduceL2,
ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd,
ReduceSum, ReduceSumSquare, Relu, Reshape, Rsqrt, ScaledTanh, Selu, Shape,
Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, Softplus, Softsign,
Split, Sqrt, Squeeze, Sub, Sum, Tan, Tanh, ThresholdedRelu, Tile, Transpose,
Unsqueeze, Xor

### TensorFlow

Even if `tract` is very far from supporting any arbitrary model, it can run
Google Inception v3 and Snips wake word models. Missing operators are easy
to add. The lack of easy to reuse test suite, and the wide diversity of 
operators in Tensorflow make it difficult to target a full support.

The following operators are implemented and tested:

Abs, Add, AddN, Assign, AvgPool, BatchToSpaceND, BiasAdd, BlockLSTM, Cast,
Ceil, ConcatV2, Const, Conv2D, DepthwiseConv2dNative, Div, ExpandDims,
FakeQuantWithMinMaxVars, Fill, FloorMod, FusedBatchNorm, GatherNd, Identity,
Less, Log, MatMul, Max, MaxPool, Maximum, Merge, Minimum, Mul, Neg, NoOp, Pack,
Pad, Placeholder, Pow, Range, RealDiv, Relu, Relu6, Reshape, Rsqrt, Shape,
Sigmoid, Softmax, SpaceToBatchND, Squeeze, StridedSlice, Sub, Switch, Tanh,
Tile, Transpose, VariableV2


### TensorFlow-Lite

TensorFlow-Lite is a TensorFlow subproject that also focuses on inference on
smaller devices. It uses a precompiler to transform a TensorFlow network to
its own format. It only supports a subset of operators from TensorFlow though,
and is only optimised for devices with Arm Neon support.

Tract supports a wider subset of TensorFlow operators, and has been optimised
for CPU of the previous generation (ARM VFP), also targetting devices in the
Raspberry Pi Zero family.

## Example of supported networks

These models among others, are used to track tract performance evolution as
part of the Continuous Integration jobs. See [.travis/README.md](readme) and 
[.travis/bundle-entrypoint.sh](.travis/bundle-entrypoint.sh) for more
information.

### Keyword spotting on Arm Cortex-M Microcontrollers

https://github.com/ARM-software/ML-KWS-for-MCU

ARM demonstrated the capabilited of the Cortex-M family by providing
tutorials and pre-trained models for keyword spotting. While the exercise
is ultimately meant for micro-controllers, `tract` can run the intermediate
TensorFlow models.

For instance, on a Rasperry Pi Zero, the "CNN M" model runs in about 70
micro-seconds, and 11 micro-seconds on a Raspberry Pi 3.

### Snips wake word models

https://arxiv.org/abs/1811.07684

Snips uses `tract` to run the wake word detectors. While earlier models were
class-based and did not require any special treatment, `tract` pulsing
capabilities made it possible to run WaveNet models efficiently enough for a
Raspberry Pi Zero.

### Inception v3

|      Device         |      Family    |  TensorFlow-lite  |  tract  |
|---------------------|----------------|-------------------|---------|
|  Raspberry Pi Zero  |    Armv6 VFP   |        113s       |   48s   |
|  Raspberry Pi 2     |    Armv7 NEON  |         25s       |    9s   |
|  Raspberry Pi 3     |  aarch32 NEON  |          5s       |    7s   |

Notes:

 * while the Raspberry Pi 3 is an Armv8 device, this bench is running
     on Raspbian, an armv6 operating system, crippling the performance
     of both benches
 * there exists other benches on the internet that show better
     performance results for TensorFlow (not -Lite) on the Pi 3.
     They use all four cores of the device. Both TensorFlow-Lite and tract
     here have been made to run on a single-core.

## Roadmap

One important guiding cross-concern: this library must cross-compile as
easily as practical to small-ish devices (think 20$ boards).

* nearly complete ONNX support, and wraps it as a backend
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

![tract-logo](assets/tract-logo/PNG/tract-horizontal-blue.png)

![rustc >= 1.41.0](https://img.shields.io/badge/rustc-%3E%3D1.41.0-brightgreen)
![MIT/Apache 2](https://img.shields.io/crates/l/tract)
[![Native Linux test status](https://github.com/snipsco/tract/workflows/Native%20Linux/badge.svg)](https://github.com/snipsco/tract/actions)
[![Embedded targets status](https://github.com/snipsco/tract/workflows/Embedded%20targets/badge.svg)](https://github.com/snipsco/tract/actions)
[![Doc](https://docs.rs/tract-core/badge.svg)](https://docs.rs/tract-core)

Sonos' Neural Network inference engine.

_This project used to be called tfdeploy, or Tensorflow-deploy-rust._

## What ?

`tract` is a Neural Network inference toolkit. It can read Tensorflow 1, ONNX
or NNEF, optimize them and run data through them.

## Quick start

* [MobileNet v2 with TensorFlow](examples/tensorflow-mobilenet-v2)
* [From Keras in Jupyter to tract](examples/jupyter-keras-tract)
* [ResNet with PyTorch](examples/pytorch-resnet)

## Tract in the landscape

### ONNX

As of today (October 2020), `tract` passes successfully about 85% of ONNX backends
tests. All "real life" integration tests in Onnx test suite are passing: 
bvlc_alexnet, densenet121, inception_v1, inception_v2, resnet50, shufflenet,
squeezenet, vgg19, zfnet512.

The following operators are implemented and tested.

Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, Cast, CategoryMapper, Ceil, Clip, Compress, Concat, Constant, ConstantLike, ConstantOfShape, Conv, ConvInteger, Cos, Cosh, DequantizeLinear, Div, Dropout, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, GRU, Gather, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HardSigmoid, Hardmax, Identity, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, MatMul, MatMulInteger, Max, MaxPool, Mean, Min, Mod, Mul, Neg, NonZero, Not, Or, PRelu, Pad, ParametricSoftplus, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, RNN, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, Round, Rsqrt, ScaledTanh, Scan, Selu, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, Softplus, Softsign, Split, Sqrt, Squeeze, Sub, Sum, Tan, Tanh, ThresholdedRelu, Tile, Transpose, Unsqueeze, Where, Xor

We test these operators against Onnx 1.4.1 (operator set 9), Onnx 1.5.0
(operator set 10), Onnx 1.6.0 (operator set 11), and Onnx 1.7.0 (operator set
12). Many networks in operator set 8 are also working.

### TensorFlow

Even if `tract` is very far from supporting any arbitrary model, it can run
Google Inception v3 and Snips wake word models. Missing operators are relatively 
easy to add. The lack of easy to reuse test suite, and the wide diversity of 
operators in Tensorflow make it difficult to target a full support.

The following operators are implemented and tested:

Abs, Add, AddN, AddV2, Assign, AvgPool, BatchToSpaceND, BiasAdd, BlockLSTM, Cast, Ceil, ConcatV2, Const, Conv2D, DepthwiseConv2dNative, Div, Enter, Equal, Exit, ExpandDims, FakeQuantWithMinMaxVars, Fill, FloorMod, FusedBatchNorm, GatherNd, GatherV2, Greater, GreaterEqual, Identity, Less, LessEqual, Log, LogicalAnd, LogicalOr, LoopCond, MatMul, Max, MaxPool, Maximum, Mean, Merge, Min, Minimum, Mul, Neg, NoOp, Pack, Pad, Placeholder, Pow, Prod, RandomUniform, RandomUniformInt, Range, RealDiv, Relu, Relu6, Reshape, Rsqrt, Shape, Sigmoid, Slice, Softmax, SpaceToBatchND, Squeeze, StridedSlice, Sub, Sum, Switch, Tanh, Tile, Transpose, VariableV2

### TensorFlow-Lite

TensorFlow-Lite is a TensorFlow subproject that also focuses on inference on
smaller devices. It uses a precompiler to transform a TensorFlow network to
its own format. It only supports a subset of operators from TensorFlow though,
and is only optimised for devices with Arm Neon support.

Tract supports a wider subset of TensorFlow operators, and has been optimised
for CPU of the previous generation (ARM VFP), also targetting devices in the
Raspberry Pi Zero family that TensorFlow Lite does not address.

### NNEF

Long story short, TensorFlow and Onnx formats are good for designing and
training networks. They need to move fast to follow the research field, tend to
integrate new features and operators greedily. They also exhibit a high level
of expressivity to facilitate network design.

On the other hand, only a subset of operators and network features actually
reach production, so systems running production network do not have to deal
with so many operators. Furthermore, some information required for training can
be stripped from the network before going to production for prediction.

NNEF tries to bridge the gap between training frameworks and inference by
proposing a format dedicated to production and prediction.

Tract supports NNEF:

* tract_nnef can load and execute NNEF networks
* tract supports most of the NNEF specification, the most notable exception
    being the ROI operators and deconvolution
* tract introduces tract-OPL, a series of NNEF extensions to support other
    operators (or extend some operators semantics) in order to represent the
    full range of tract-core neural network support: any network understood by
    tract should be serializable to tract-OPL. This is a work in progress.
* tract command line can translate networks from TensorFlow or ONNX to NNEF/OPL.

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
|  Raspberry Pi Zero  |    Armv6 VFP   |        113s       |   39s   |
|  Raspberry Pi 2     |    Armv7 NEON  |         25s       |    7s   |
|  Raspberry Pi 3     |  aarch32 NEON  |          5s       |    5s   |

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

Note: files in the `tensorflow/protos` directory are copied from the
[TensorFlow](https://github.com/tensorflow/tensorflow) project and are not
covered by the following licence statement.

Note: files in the `onnx/protos` directory are copied from the
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

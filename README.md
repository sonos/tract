![tract-logo](assets/tract-logo/PNG/tract-horizontal-blue.png)

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![rustc >= 1.75.0](https://img.shields.io/badge/rustc-%3E%3D1.75.0-brightgreen)
![MIT/Apache 2](https://img.shields.io/crates/l/tract)
[![Native Linux test status](https://github.com/snipsco/tract/workflows/Native%20Linux/badge.svg)](https://github.com/snipsco/tract/actions)
[![Embedded targets status](https://github.com/snipsco/tract/workflows/Embedded%20targets/badge.svg)](https://github.com/snipsco/tract/actions)
[![Doc](https://docs.rs/tract-core/badge.svg)](https://docs.rs/tract-core)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://pypi.org/project/tract/)


Sonos' Neural Network inference engine.

_This project used to be called tfdeploy, or Tensorflow-deploy-rust._

## What ?

`tract` is a Neural Network inference toolkit. It can read ONNX or NNEF, optimize them and run them.

## Quick start, examples

* [MobileNet v2 with ONNX](examples/onnx-mobilenet-v2)
* [BERT example with ONNX](examples/pytorch-albert-v2)
* [MobileNet v2 with TensorFlow](examples/tensorflow-mobilenet-v2)
* [From Keras and TensorFlow 2 to tract](examples/keras-tract-tf2)
* [ResNet with PyTorch](examples/pytorch-resnet)

There is also [some technical documentation](doc/) and [blog](https://tech-blog.sonos.com/posts/optimising-a-neural-network-for-inference/) posts.

## Tract in the landscape

### ONNX

As of today, `tract` passes successfully about 85% of ONNX backends
tests. All "real life" integration tests in ONNX test suite are passing: 
bvlc_alexnet, densenet121, inception_v1, inception_v2, resnet50, shufflenet,
squeezenet, vgg19, zfnet512.

Notable missing parts are operators dealing with Tensor Sequences and Optional Tensors : tract /really/ wants to flow Tensors and nothing else.
This is structural. Changing it would be pretty difficult, and it's unclear whether it can be done without impairing performance or maintainability.
We are not convinced these features have shown their interest in the wild yet, so we prefer to leave them aside.

Other dark corners are specific operators like "Resize" which fit perfectly in the framework but need a complex internal logic that is far
from our core business. In these cases, we are happy to accept contributions and to help. 

The following operators are implemented and tested.

Abs, Acos, Acosh, Add, And, ArgMax, ArgMin, ArrayFeatureExtractor, Asin, Asinh, Atan, Atanh, AveragePool, BatchNormalization, BitShift, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor, BlackmanWindow, Cast, CastLike, CategoryMapper, Ceil, Clip, Compress, Concat, Constant, ConstantLike, ConstantOfShape, Conv, ConvInteger, ConvTranspose, Cos, Cosh, CumSum, DFT, DepthToSpace, DequantizeLinear, Div, Dropout, DynamicQuantizeLinear, Einsum, Elu, Equal, Erf, Exp, Expand, EyeLike, Flatten, Floor, GRU, Gather, GatherElements, GatherND, Gemm, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, Greater, GreaterOrEqual, HammingWindow, HannWindow, HardSigmoid, Hardmax, Identity, If, InstanceNormalization, IsInf, IsNaN, LRN, LSTM, LeakyRelu, Less, LessOrEqual, Log, LogSoftmax, MatMul, MatMulInteger, Max, MaxPool, Mean, MelWeightMatrix, Min, Mod, Mul, Multinomial, Neg, NonMaxSuppression, NonZero, Not, OneHot, Or, PRelu, Pad, ParametricSoftplus, Pow, QLinearConv, QLinearMatMul, QuantizeLinear, RNN, RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike, Range, Reciprocal, ReduceL1, ReduceL2, ReduceLogSum, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, ReduceSumSquare, Relu, Reshape, Resize, Round, Rsqrt, STFT, ScaledTanh, Scan, Scatter, ScatterElements, ScatterND, Selu, Shape, Shrink, Sigmoid, Sign, Sin, Sinh, Size, Slice, Softmax, Softplus, Softsign, SpaceToDepth, Split, Sqrt, Squeeze, Sub, Sum, Tan, Tanh, ThresholdedRelu, Tile, Transpose, TreeEnsembleClassifier, Unsqueeze, Where, Xor

We test these operators against from ONNX 1.4.1 (operator set 9), up to ONNX 1.13.0 (operator set 18).

We are using ONNX test suite, but it does not cover everything.
We also deliberately ignore some tests, or restricting their scope depending on what we feel is realistic.
Sometimes these decisions are just wrong, and sometimes they become wrong as time goes by and the fields moves in unexpected directions.
So if you are puzzled by an ONNX model that does not work in tract, we are happy to take a look.

### NNEF

Long story short, TensorFlow and ONNX formats are good for designing and
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
    being the ROI operators
* tract introduces tract-OPL, a series of NNEF extensions to support other
    operators (or extend some operators semantics) in order to represent the
    full range of tract-core neural network support: any network understood by
    tract should be serializable to tract-OPL. This is a work in progress.
* tract command line can translate networks from TensorFlow or ONNX to NNEF/OPL.

### tract-opl version compatibility

A remainder: NNEF is not expressive enough to represent all ONNX. tract-OPL extends
NNEF using proprietary to support what is missing. Notable extensions are pulse
operators, recurring operators (as Scan) and symbolic extensions.

There is no strict check in place here, so... implementation is not bullet proof.
* NNEF part aims at being very stable. It is strongly constrained with compatibility
with NNEF specification.
* tract-opl is a bit more in flux. Nevertheless we try to maintain the following
golden rule:

     `models serialized with tract 0.x.y should work with tract 0.x.z where z >= y`

* in practice, breaking changes have been relatively rare so far. Most models are
forward and retro compatible from when tract has acquired NNEF support.

Notable breakage occurred:
* 0.16.3 (forward compatible) on Scan operator
* 0.17.0 for binary decision tree classifier

Starting with `0.17.0`, a model property is injected in tract-opl files (`tract_nnef_ser_version`)
to tag which version of tract generated the file. As most models will remain compatible,
tract will not do any version check. It is up to the application developer to do so.

A softer version tag exists as `tract_nnef_format_version`. pre-0.17.0 version set it to
`alpha1`, post-0.17.0 set it `beta1`. Don't put too much emphasis into the "alpha-ness" naming 
of versions here.

### Note: support for TensorFlow 1.x

Even if `tract` is very far from supporting any arbitrary model, it can run
Google Inception v3 and Snips wake word models. Missing operators are relatively 
easy to add. The lack of easy to reuse test suite, and the wide diversity of 
operators in Tensorflow make it difficult to target a full support.

The following operators are implemented and tested:

Abs, Add, AddN, AddV2, Assign, AvgPool, BatchToSpaceND, BiasAdd, BlockLSTM, Cast, Ceil, ConcatV2, Const, Conv2D, DepthwiseConv2dNative, Div, Enter, Equal, Exit, ExpandDims, FakeQuantWithMinMaxVars, Fill, FloorMod, FusedBatchNorm, GatherNd, GatherV2, Greater, GreaterEqual, Identity, Less, LessEqual, Log, LogicalAnd, LogicalOr, LoopCond, MatMul, Max, MaxPool, Maximum, Mean, Merge, Min, Minimum, Mul, Neg, NoOp, Pack, Pad, Placeholder, Pow, Prod, RandomUniform, RandomUniformInt, Range, RealDiv, Relu, Relu6, Reshape, Rsqrt, Shape, Sigmoid, Slice, Softmax, SpaceToBatchND, Squeeze, StridedSlice, Sub, Sum, Switch, Tanh, Tile, Transpose, VariableV2

Additionally, the complexity of TensorFlow 2 make it very unlikely that a direct
support will ever exist in tract. But many TensorFlow 2 models can be
converted to ONNX and then loaded in tract.

## Example of supported networks

These models among others, are used to track tract performance evolution as
part of the Continuous Integration jobs. See [.travis/README.md](readme) and 
[.travis/bundle-entrypoint.sh](.travis/bundle-entrypoint.sh) for more
information.

### Keyword spotting on Arm Cortex-M Microcontrollers

https://github.com/ARM-software/ML-KWS-for-MCU

ARM demonstrated the capabilities of the Cortex-M family by providing
tutorials and pre-trained models for keyword spotting. While the exercise
is ultimately meant for micro-controllers, `tract` can run the intermediate
TensorFlow models.

For instance, on a Raspberry Pi Zero, the "CNN M" model runs in about 70
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

# License

Note: files in the `tensorflow/protos` directory are copied from the
[TensorFlow](https://github.com/tensorflow/tensorflow) project and are not
covered by the following licence statement.

Note: files in the `onnx/protos` directory are copied from the
[ONNX](https://github.com/onnx/onnx) project and are not
covered by the following license statement.

## Apache 2.0/MIT

All original work licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.

## Contribution

Unless you explicitly state otherwise, any Contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.

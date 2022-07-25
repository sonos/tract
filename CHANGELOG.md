# Unreleased

# 0.17.3 - 2022-07-25
* [License] Allowing https://spdx.org/licenses/Unicode-DFS-2016.html (no tldr yet, but pretty similar to BSD-2)
* [Breaking] CLI --json option reports costs as strings instead of numbers (in order to allow symbolic values).
* Sigmoid/Tanh f32 reimpl, plus new f16 impl.

# 0.17.1 - 2022-07-11
* Sanitiser=address in the CI. Fixed a couple of overflow/memleaks. (Nothing looked too awful.)
* ONNX NonMaxSuppression

# 0.17.0 - 2022-06-13
* [Breaking] [ONNX-ML] TreeEnsembleClassifier with binary output (single class) now mimicks scikit-learn output layout.

# 0.16.9 - 2022-06-10
* bump ONNX protobuf file and support external tensors format
* new "skinny" kernels for avx2/fma f32 multiplication (positive impact on low, non 1 batch size for DNN-heavy loads)

# 0.16.7 - 2022-05-16
* Softmax is now an operator in core, coming with a direct quantized implementation
* new TypedFact constructor API ( f32::fact(&[1, 4, 12]), f32::fact(shape!(Symbol::from('N'), 12)))
* fixes and optimisation of re-quantization pipeline
* fixes around symbols in NNEF/OPL

# 0.16.6 - 2022-05-03
* Various changes around quantization support (qi32 appearance)

# 0.16.5 - 2022-04-27
* Intel optimisation are back
* Range is now more flexible, should unlock some BERT models with symbolic dimensions.

# 0.16.4 - 2022-04-14
* some optimisations in depthwise convolutions
* various bugfixes
* [Breaking] Fixed nnef "tile" operator definition ("repeats" is plural). As a consequence models using "tile" serialized with tract with prior versions can not be loaded anymore (and vice-versa).

# 0.16.3 - 2022-03-30
* [Breaking] tract-opl models Scan syntax changed a bit. Models exported by <0.16.2 are loadable in >=0.16.2, but not the other way around.
* Optimisation in deconv

# 0.16.1 - 2022-03-02
* [Breaking] Minimum Rust Supported Version is now 1.59.0
* [Breaking] Small API changes in model api: .compact(), .optimize(), .declutter() now take &mut self and work in place.
* [LICENSE] Only the licensing for dependencies of the top-level library crates (tensorflow, onnx, kaldi, pulse) will now be monitored. The command line tool (tract crate in cli folder) is for developpers (tract developpers or tract integrators), is not meant to be shipped to end-user, and it concentrates most of the license and dependency complexity.
* [LICENSE] BSD-3-Clause is now accepted in tract.
* Optimisations around convolutions and deconvolution
* Optimisation on Cortex-A53, first round of Cortex-A55 optimisation too.

# 0.15.8 - 2021-11-18
* Fix brand new ArrayFeatureExtractor inference

# 0.15.7 - 2021-11-16
* ONNX ArrayFeatureExtractor
* ConvTranspose/deconv optimisation

# 0.15.6 - yanked 
* just a release script failure

# 0.15.5 - 2021-10-26
* hold half at 1.7.x for compat with rust 1.50

# 0.15.4 - 2021-10-21
* ConvTranspose/deconv pulse support
* ONNX SpaceToDepth/DepthToSpace

# 0.15.3 - 2021-07-29
* optimise i8*u8, u8*i8 and u8*u8 matrix products (and convo)

# 0.15.2 - 2021-07-09
* bump prost dep

# 0.15.1 - 2021-07-08
* some optimisations for arm32 (cortex-a7 and a9)

# 0.15.0 - 2021-06-24

* Switched the order of item_type and item_type_vendor in the NNEF tendor format to be consistent with NNEF-tools, and changed the item_type of integers due to an error in the specification. Breaking for tensor files containing integers or strings.
* Scan output batching optimisation
* Concat pulsification over a secondary axis
* new aarch64 16x4 f32 kernel

## 0.14.2 - 2021-05-27

* better handling of errors in ONNX parser
* fix/workaround some performance regressions bubling from recent ndarray changes

## 0.14.1 - 2021-05-18

* ONNX ConvTranspose, Gather, GatherND, GatherElements, Scatter, ScatterND, ScatterElements support (and NNEF deconv)
* Fixes around integer serialisation in NNEF
* workaround subtle breaking changes in ndarray (between 0.15.1 and 0.15.2)

## 0.14.0 - 2021-04-19

* low-level functions in linalg are now version tagged: two versions of tract can now co-exist in the same binary
* rustc minimal version is now 1.50
* dependencies version bumps (ndarray, itertools, and others)

## 0.13.2

* fix sigmoid and tanh variability on intel

## 0.13.1

* temporary disable binary unicast add fusing (too many bugs)

## 0.13.0

* Release are now "in sync": all tract crate versions on a build *must* be aligned
* optimisations, with a focus on aarch64

## 0.12.5 - 2021-01-12

* Dependency bumps

## 0.12.4 - 2021-01-06

* 0.12.3 is a misfire
* hotfixes on 0.12.2 new tree classifier
* fix X compilation from macos/aarch64 to macos/intel

## 0.12.2 - 2021-01-05

* ONNX-ML: CategoryMapper and TreeEnsembleClassifier (partial, SoftmaxZero and Probits are missing). With NNEF support.
* cargo-deny enforces licences choices

## 0.12.1 - 2020-12-11

* 0.12.0 is a misfire.

* API BREAKING: TypedFact::dt_shape & friends can not fail anymore, no longer return a result (remove `?`)
* Breaking: Rust minimal version bumped to 1.42

* Early, basic, correct but slow support for i8 by u8 matrix mult.
* Support for Apple Silicon, aka M1, aka aarch64 darwin (but not in CI yet)
* dynamic quantization convolution support
* release now ships cli musl builds for linux
* optimizations targetting small Cortex-A (like 7, 8, and 9)
* command line dump --profile --cost now computes flops
* ONNX: OneHot op support

## 0.11.2 - 2020-10-26

* ONNX: new op: DynamicQuantizeLinear
* tract-data crate split from core, containing tensor, dim, and datum types.

## 0.11.1 - 2020-10-20

* switch from error_chain to anyhow
* simplify trivial gathers to a slice
* generalize symbolic dimension a bit: support "2S" and the like
* deprecate "x" syntax in CLI, please use `,`  instead

## 0.11.0

### Breaking 

* NNEF: tract-nnef no longer performs gunziping, but expect an uncompressed tar
    stream. We found out is it counter-productive (weights matrices are more or
    less random, they do not compress easily, and decompression is expensive).
    NNEF networks in the wild are .tgz file. Using flate2, decompression is a
    one-liner, but it must be done by the client code now.
* bumped extended nnef compat version (unchecked at this stage) to "alpha1"
* move pulse operators and translation to their own crate and nnef registry
* generalize TDim to support an arbitrary number of symbols
* concretize_stream_dim is superseded by concrentize_dims

### Notable

* new crates, building on tract-opl introduction:
    * *tract-pulse-opl*: pulse runtime (handful of ops, including Delay) is now separated from core
    * *tract-onnx-opl*: onnx runtime (4 ops not belonging in core)
    * *tract-pulse*: pulsification of models (model-translation time)
    * tract-onnx is now limited to onnx model loading and conversion

## 0.10.10 - 2020-08-30

* load a NNEF as a TypedModel using tract_nnef, and from the CLI
* dump a tract TypedModel to NNEF (with extensions for op not nnef compatbile)
* not a full coverage of nnef, but enough for most CNN (image categorizers zoo working)
* 80% of onnx tests are surviving a NNEF dump and reload at this stage

## 0.10.0 - 2020-07-28

### ONNX

* covered operators compatible with Operator Sets 9, 10, 11 (new) and 12 (new)

### API Breaking

* Tensor::l1 method is gone

### Windows

* Support for -gnu targets (non-mvsc).

### Notable

* --cost now gives the number of parameters in the model
* SimpleState is clonable again (actually useful !)

## 0.9.2 - 2020-06-16

* introduce `TypedModel::method.concretize_stream_dim`
* various pulsification bugfixes

## 0.9.1 - 2020-06-16

* fix Reshape with TDim

## 0.9.0 - 2020-06-15

Still no shortage of version numbers...

### API Breakage

* NormalizedModel (and friends) are gone. They were only useful as a pre-pulse transformation pre-requisite that the current TypedModel (& co) meets.
* TypedModel::into_optimized() is gone. InferenceModel::into_optimized() stays as an end-to-end shortcut for simple cases. It does .into_typed()?.declutter()?.optimize()).
* TypedModel::codegen() is now ::optimize()

## 0.8.0 - 2020-06-13

I wish I had seen these issues yesterday. Anyway, version numbers are cheap.

* Bumping minimum rust to 1.41

## 0.7.0 - 2020-06-12

* CLI refactoring (hopefully stabilizing a bit?)
    * `profile --bench` is now bench
    * profile is now `dump --profile`
    * cost is now `dump --cost`
    * profiling is now done during a full net instead of per op
    * new "compact" graph dumper, profile visual hints
    * `dump --cost --profile --json` output profiling and cost information
    * show logical names for ops instead of the Op struct names (not 100% sure it's right)
    * criterion integration
* WASM support for tract-onnx and tract-tensorflow targets (CI)
* Convenience methods added to Models to allow model building in fluent style, up to Plan instantiation (SimplePlan now nicknamed RunnableModel). Non breaking.
* Support for ONNX bidi LSTM (CI), GRU and RNN (untested, consider alpha)
* Fixes around nets with a non trivial batch size (axis simplification code, matmul op fusion)

## 0.6.3 - 2020-04-25

* Lock ndarray version to dodge rustc/llvm issue (https://github.com/rust-lang/rust/issues/71506)

## 0.6.2 - 2020-04-15

* Use http://gihub.com/kali/readings for instrumentation.

## 0.6.0 - 2020-02-19

### Notable

* New jupyter/keras/tf example
* ARMv8 tanh / sigmoid optimisation

### API Breaking

* refactor exports and dependencies
    * preferred way to use tract is now to `use tract_tensorflow::prelude::*;`
    * singleton framework is built by `let tensorflow = tensorflow()`. The Framework trait is in the prelude too.
    * the prelude contains a reexport of `tract_core`, and of ndarray as `tract_ndarray`
    * no more need to declare dependency on `tract-core` and/or `tract-linalg` in Cargo.toml
    * same goes for `tract_onnx`

## 0.5.9 - 2020-02-07

### Breaking

* Rustc minimum version is now 1.39

### Onnx

* Support for MatMulInteger, ConvInteger
* Support for QuantizeLinear DequantizeLinear
* Basic support for QLinearMatMul, QLinearConv

## 0.5.6 - 2019-10-30

### Tensorflow

* Initial support for GatherV2

### Onnx

* Fix PReLu normalization

## 0.5.5 - 2019-10-25

### Tensorflow

* Initial support for AddV2, Mean, Min, Prod, Sum

## 0.5.4 - 2019-09-30

### Notable

* Make Onnx loader operator set aware, and Slice-10 support.
* Cost now reports Delay ops buffer size
* Bump dependencies (protobuf) and fix codegen
* Windows CI now performs a top-level "cargo check"

## 0.5.1 - 2019-09-24

### Bugfix

* remove the no_panic checks, as too fragile (breaking non-lto builds)

## 0.5.0 - 2019-09-20

### Breaking

* Change tensor facts names for consistency: TensorFact is now InferenceFact.

### Notable

* Introduce Windows support, including CI coverage for linalg
* Switch from Travis to GitHub Actions
* Internal refactoring around tract-core canonic opset
* Tract CLI can now compute a FLOP number for networks ("cost" subcommand). 
    Furthermore the CI asserts its value for a few networks to prevent optimisation regressions.
* Fix: handling of -1 in ONNX Reshape op

## 0.4.2 - 2019-09-10

* Fix release script after 0.4.1 release disaster.

## 0.4.1 - 2019-09-09 [YANKED]

* Fix for OS where CARGO_CFG_TARGET_FAMILY is undefined
* Linear Algebra package refactor
* tract-core canonic operator set introduction
* significant performance boost (up to 20% on some real-life networks)

## 0.4.0 - 2019-07-30

* Start Kaldi networks support (LSTM, Renorm, Affine, downsample)

## Before...

This Changelog started way too late. But better late than never.


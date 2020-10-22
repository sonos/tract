## Unrelased

* ONNX: new op: DynamicQuantizeLinear

## 0.11.1 - 2020-10-20

* switch from error_chain to anyhow
* simplify trivial gathers to a slice
* generalize symbolic dimension a bit: support "2S" and the like
* deprecate "x" syntax in CLI, please use , instead

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


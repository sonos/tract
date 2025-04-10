# Unreleased

# 0.21.12 - 2025-04-10

* multithread matmul is feature gated now ("multithread-mm" on linal)
* full hand made arm64 f32-accumulator matmul kit
* more auditing improvment around einsum and its matmul translations
* bugfix in matmul translation and gather
* more test-rt-level coverage of low-level matmuls (metal and cpu)
* memory arena improvements (metal)
* q40 for convolution weights


# 0.21.11 - 2025-03-19

* [cli] augment audit capabilities for mm implementation choices
* revisit matmul kernel selection
* improve gather with compressed inputs
* revisit slice bubbling up to unlock optimisations
* fix a bug around flipping substractions
* support for left q40 input in arm64 f32 accumulating kernels (unlocks q40f32 compression on arm64)

# 0.21.10 - 2025-02-21
* WIP llm testability (--approx-custom)
* [metal] ggml-ported kernels
* WIP einsum-to-matmul testability
* optimisation around reduce<sum> impacting some modern/exotic normalisation layers
* WIP towards better handling of shared weights (e.g. embeddings duplication)

# 0.21.9 - 2025-01-08
* [metal] experimental profile
* [cpu] new versatile (mmm/mmmv) kernels combinations for various architectures
* [metal] scaled-masked-softmax detector and impl


# 0.21.8 - 2024-12-05
* [linalg, compression] introduce mmm kits
* [linalg] (wip) rework f16 on non-f16 machines
* [linalg] element-wise binary operators optimisation
* [core, compression] gather with compressed weights
* [metal] new kernels
* [metal] new memory management
* [nnef] opt-in deterministic tar encoding

# 0.21.7 - 2024-09-23
* [metal] (experimental) introduce partial support for Apple Metal
* [core] Potential internal API breaking changes (operator names, comparison ops refactored)
* [data] (experimental) Smarter TDim simplification, handling of Min and Max. TDim assertions for simplifications.
* [data] (experimental) WIP around multiple scenarios (modes) for LLM inference
* Extra examples
* [linalg] (experimental) kernels targetting LLM block-quantized tasks (inc. intel 32x1 q40f32)

# 0.21.6 - 2024-07-24
* [data] Rework tdim and symbols, introduce inequalities assertions, min and max operators
* [data] Generalize Blob usage in Tensor
* [linalg] Rework reduce implementation, introduce more generic binary ops support (wip)
* [linalg] Introduce multithreaded matrix multiplication runner
* [linalg] Introduce Q4_0 block quantization for weights (wip)
* [linalg] Introduce AMX f16 kernels, Neon Q40F16 kernel (experimental)
* [linalg] wasm f32 4x4 kernel 
* [core] Introduce Opaque and OpaqueFact to escape Tensor and TValue formalism
* [core] generalize/improve float precision translator, with translation filter
* [core] Introduce garbage collecting in patch application, new compact algo, and rework constant propagation to spare memory
* [core] Rework packed format and packing metadata
* [linalg/core] Introduce multiple packing format for matmul kernels
* [core] Work In Progress refactoring binary, towards more optimized execution strategies
* [nnef] inequalities assertions extension, q4_0 extension
* [tflite] plug in tanh and sigmoid

# 0.21.5 - 2024-05-11
* [TFLite] fixes for fully connected and max pool layers
* Allow opting out of new memory friendly execution order optimisation

# 0.21.4 - 2024-04-23
* More memory/cache friendly execution order
* Several fixes around symbolic dimensions computation (some should help with attention models)

# 0.21.3 - 2024-04-03
* [AMX] Put AMX for iOS behind a feature gate ("tract-linalg/apple-amx-ios").

# 0.21.2 - 2024-03-29 (yanked)
* [ONNX] Support for external storage of tensors with offset and length
* [ONNX] Lots of fixes around binary quantized operators (add, mul, etc)
* [PY] Fix python source distribution
* [AMX] Activate AMX on iOS
* [API] Introduce transforms in external api
* [BLAS] Introduce a simple BLAS transform for Matrix multiplication
* [F16] Introduce a Reduce<MeanOfSquares> that solves many L2 normalization errors in f16

This version has been yanked to revert systematic activation of AMX on iOS. AMX is a private API and Apple may reject an App that performs AMX instructions.

# 0.21.1 - 2024-02-08
* [ONNX] Support for external storage of tensors with offset and length

# 0.21.0 - 2024-01-16
* MSRV is now 1.75.0
* [internal] ConvUnary and MatmulUnary are replaced by binary, potentially dynamic equivalent

# 0.20.22 - 2023-11-28
* [ONNX] LayerNormalization support

# 0.20.21 - 2023-10-31
* [ONNX] ignoring output shapes is now the default
* 

# 0.20.18 - 2023-08-30
* [intel] fix in AVX512F matrix vector product
* [tflite] alpha, embryonic support. some convolutional models working.
* [kaldi] remove abandonned kaldi experimental support
* [refactoring] Runtime abstraction and runtime-targetting tests
* [refactoring] Refactoring Python and C API around a possible tract-api. Introducing dylib support.
* [pytorch compat] fixes around node names starting by / (bug triggered by recent pytorch versions)

0.20.7 to 0.20.17 are misfires

# 0.20.6 - 2023-06-09
* Bug fixes, fix display of If operator

# 0.20.5 - 2023-05-26
* Various bugfix around Einsum
* Einsum now has functions to translate to MatMul and other axes manipulations

# 0.20.0, 0.20.1, 0,20.2, 0.20.3 - 2023-04-25
* [optim] 32x32 f32 AMX kernel (for Apple Silicon M family)
* [optim] bunch of AMX512F kernels (square, skinny, vector)
* [ONNX] introduce Trilu, TopK
* [NNEF/OPL] submodel loader
* [ONNX] support alternative layout for LSTM (layout=1, batch becomes first axis)
* [ONNX] If operators with dynamic condition (very basic optimisations, no nnef support yet).

# 0.19.9 & 0.19.10 - 2023-04-17
* HardSwiwh ONNX, tract_core_hard_swish in NNEF/OPL
* introducing tract_core_submodel in NNEF/OPL
* JSON resource loader in NNEF/OPL
* Profiling API tweaks
* `--folded` view for model command line dump (hide Scan loops)

# 0.19.8 - 2023-03-27
* Various bug fixes

# 0.19.7 & 0.19.6 - 2023-02-24
* more bug fixes
* wip on python doc auto-deploy

# 0.19.5 - 2023-02-22
* 0.19.3 and 0.19.4 are release misfires
* lots of bugfixes following 0.19 big changes
* introducing the JSON NNEF resource

# 0.19.2 - 2023-01-30
* [NNEF/OPL] introduce json resource loader
* extend Complex number support (under a feature flag)

# 0.19.1 - 2023-01-23
* [nnef] new identifier syntax is now opt-in for serialization (both accepted at loading)
* alpha-level C interface. how and how to deploy it (where to put the .h, whether or not to build and ship dylibs)
* alpha-level python interface. deployed on pypi as "tract". At this stage, API is undocumented and may still change significantly.

# 0.19.0 - 2023-01-11
* [BREAKING] TValue are now used in run() instead of the previous mix of Tensor and Arc<Tensor>
* internal API breaking changes: no more op_families, libcli split away. State is no longer Send (but can be "frozen" to a Send counterpart).
* Symbols can now be String instead of char. They are not shared globally anymore, but scoped in the Model instead.
* [pulse] S symbol is no longer magic. The time dimension symbol must be provided at pulsification time.
* [pulse] In most cases, we can now pulsify without an explicit pulse len (pulse len can be expression).
* [cli] deprecated "x" syntax for shape is removed
* [nnef/opl] new i"..." syntax for escaping identifiers: i"some arbitrary string". Allow serialization of any ONNX model with any kind of string as node names.
* [ONNX] Signal processing operators (DTF, STFT, MelWeightMatrix, BlackmanWindow, HammingWindow, HannWindow)
* [ONNX] bitwise operations
* [ONNX] Compatibility target raised to operator set 18

# 0.18.3 - 2022-10-27
* [NNEF] Introduce a "resource" extension for loading values from a separate source (as a config file)
* Workaround for cpu detection failure on FreeBSD / arm64
* Various bug fixes

# 0.18.2 - 2022-10-18
* [pulse] improve convolution (and others) pulsification to avoid some unecessary buffering delay
* [cli] support multiple streaming inputs and outputs
* [ONNX] more relaxed Clip operator rules

# 0.18.1 - 2022-10-06
* prepare NNEF for further tract-opl extension (resource support)
* more generic matmul
* optimise some EinSum cases as matmul

# 0.18.0 - 2022-09-21
* [ONNX Breaking] Several changes to move towards supporting ONNX symbolic dimensions (actual fixes, but they may break stuff that was working more or less by accident). It may be required to erase output shapes explicitely when input shape is overriden on models that were working before.
* [CLI breaking] ONXN symbolic dimensions has some impact here too. --input-bundle is deprecated, is was overriden and ambiguous. Instead, there is a  --input-facts-from-bundle global option, and a --input-from-bundle option in the subcommands run, profile, dump. --allow-random-input is also moved to subcommands. We think all previously supported behaviours are still there. Please open issues if not.

# 0.17.7 - 2022-09-05
* clippy up all tract code
* various fixes
* 0.17.5 and 0.17.6 are misfired

# 0.17.4 - 2022-08-11
* [cli] global --set (as a somehat cleaner --concretize successor) allow to set a symbol value after decluttering
* [cli] run --save-outputs output.npz to save execution outputs
* dozens of fixs and code cleanup (clippy-fication in progress)

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


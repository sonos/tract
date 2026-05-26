![tract-logo](assets/tract-logo/PNG/tract-horizontal-blue.png)

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![rustc >= 1.91.0](https://img.shields.io/badge/rustc-%3E%3D1.91.0-brightgreen)
![MIT/Apache 2](https://img.shields.io/crates/l/tract)
[![Native Linux test status](https://github.com/sonos/tract/workflows/Native%20Linux/badge.svg)](https://github.com/sonos/tract/actions)
[![Embedded targets status](https://github.com/sonos/tract/workflows/Embedded%20targets/badge.svg)](https://github.com/sonos/tract/actions)
[![Doc](https://docs.rs/tract-core/badge.svg)](https://docs.rs/tract-core)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://pypi.org/project/tract/)

Sonos' neural-network inference engine.

tract loads ONNX and NNEF models, optimises them, and runs them anywhere —
from embedded ARM CPUs to NVIDIA / Apple GPUs, in the browser via
WebAssembly, or on a Linux / macOS / Windows workstation. It is used in
production at Sonos for wake-word and streaming speech-recognition
workloads, and also runs LLM, text-to-image, and classical CV models with
a particular focus on the *translate-once / ship-tiny-runtime* story
enabled by its NNEF-based intermediate format (tract-OPL).

## Quick start

From [`examples/onnx-mobilenet-v2`](examples/onnx-mobilenet-v2):

```rust
use tract::prelude::*;
tract::impl_ndarray_interop!();

let model = tract::onnx()?
    .load("mobilenetv2-7.onnx")?
    .into_model()?;

// prepare() optimises and compiles the model for the chosen runtime
let runtime = tract::runtime_for_name("default")?;
let runnable = runtime.prepare(model)?;

let result = runnable.run([input.tract()?])?;
```

The [`tract`](https://crates.io/crates/tract) crate (`api/rs/src/lib.rs`) is the authoritative public API. The
internal crates (`tract-core`, `tract-nnef`, `tract-onnx`, ...) are not
stable surface and shouldn't be depended on directly.

For Python, see the [`tract`](https://pypi.org/project/tract/) package on PyPI.

## Examples

[`examples/`](examples/) has runnable demos covering the workloads tract
targets today:

| Example | What |
|---|---|
| [`onnx-mobilenet-v2`](examples/onnx-mobilenet-v2) | Minimal CV starter |
| [`tflite-mobilenet-v3`](examples/tflite-mobilenet-v3) | TFLite import path |
| [`causal_llm`](examples/causal_llm) | Transformer text generation |
| [`nemo-parakeet-asr`](examples/nemo-parakeet-asr) / [`nemo-nemotron-streaming-asr`](examples/nemo-nemotron-streaming-asr) | Speech recognition, including streaming via pulsification |
| [`stable-diffusion`](examples/stable-diffusion) / [`stable-diffusion-3`](examples/stable-diffusion-3) / [`stable-diffusion-xl`](examples/stable-diffusion-xl) | Text-to-image |
| [`face_detection_yolov8onnx_example`](examples/face_detection_yolov8onnx_example) / [`face_similarity_arcface_onnx`](examples/face_similarity_arcface_onnx) | Modern object detection / face recognition |
| [`wasm-model-bench`](examples/wasm-model-bench) | Running tract in the browser |

## Resources

Technical documentation lives under [`doc/`](doc/) (start at [`doc/intro.md`](doc/intro.md));
the [`doc/cli-recipe.md`](doc/cli-recipe.md) page collects practical CLI recipes.
The Sonos engineering [blog](https://tech-blog.sonos.com/posts/optimising-a-neural-network-for-inference/)
has a long-form post on tract internals.

## Python

tract is also available as the [`tract`](https://pypi.org/project/tract/) package on PyPI,
built on top of the same Rust core:

```sh
pip install tract
```

The API mirrors the Rust pipeline: load a model, set input facts, optimise, then run.
Documentation: [sonos.github.io/tract](https://sonos.github.io/tract). Source lives in [`api/py/`](api/py/).

## Runtimes

| Runtime | Name | Crate | Notes |
|---|---|---|---|
| CPU (x86, ARMv6/7/8, ARM SVE) | `"default"` | `tract-linalg` | Default. Hand-rolled SIMD micro-kernels. |
| Apple Metal | `"metal"` | `tract-metal` | Apple GPUs. |
| NVIDIA CUDA | `"cuda"` | `tract-cuda` | NVIDIA GPUs. |
| WebAssembly | `"default"` | _via standard wasm32 targets_ | Browser / WASI deployment. |

All runtimes share the `TypedModel` IR and the same loaders, so a model
optimised on one platform can be moved to another.

## Streaming and pulsification

tract has first-class support for *pulsified* inference: a network that
operates on full sequences during training is translated into one that
processes a fixed-size pulse along its streaming axis at each step. This
lets the same model serve both batch evaluation and low-latency real-time
inference (wake-word, streaming ASR, ...).

The translate-time logic lives in `tract-pulse`; runtime ships only the
small `tract-pulse-opl` crate. See
[`AGENTS.md` § Streaming and pulsification](AGENTS.md#streaming-and-pulsification)
for the engineering view, and
[`examples/nemo-nemotron-streaming-asr`](examples/nemo-nemotron-streaming-asr)
for a working demo.

## Formats and tract-OPL

| Format | Load | Save |
|---|---|---|
| ONNX | ✓ | — |
| NNEF (+ tract-OPL extensions) | ✓ | ✓ |
| TensorFlow Lite (legacy) | ✓ | ✓ |
| TensorFlow 1 frozen graph (legacy) | ✓ | — |

PyTorch models can be exported directly to NNEF using
[torch-to-nnef](https://sonos.github.io/torch-to-nnef/latest/)
([source](https://github.com/sonos/torch-to-nnef)), an open-source
PyTorch-to-NNEF converter maintained alongside tract — useful when you want
to skip the detour through ONNX.

tract-OPL is an NNEF-compatible intermediate representation that extends
NNEF with the operators needed to express a full tract-core model. The
recommended deployment workflow is:

1. **Once, at build time:** convert from ONNX / TF / TFLite to NNEF using
   the `tract` CLI:
   ```sh
   tract model.onnx dump --nnef model.nnef.tgz
   ```
2. **At runtime:** ship only `tract-core` + `tract-nnef`, plus
   `tract-onnx-opl` if the model uses ONNX-only operators, and
   `tract-pulse-opl` if it is pulsified.

This keeps the runtime footprint small (no protobuf, no training-framework
loaders). See [`doc/intro.md`](doc/intro.md) for the full design rationale.

### tract-OPL stability

NNEF parts are tied to the NNEF specification and very stable. tract-OPL
extensions are a bit more in flux, but we observe the rule:

> A model serialised with tract `0.x.y` should work with tract `0.x.z` where `z >= y`.

Models embed a `tract_nnef_ser_version` property identifying the generating
tract version; tract itself does not enforce a version check, so it is up
to the application to do so if needed. See [`CHANGELOG.md`](CHANGELOG.md)
for the running list of notable serialisation-format changes.

## TensorFlow 1 (legacy)

tract still loads TF1 frozen graphs and supports the operator set needed
for the classical CV and wake-word models that originally drove its design
(Inception v3, Snips wake words, ...). TensorFlow 2 is not directly
supported — convert to ONNX first.

## License

Files in `tensorflow/protos` are copied from the
[TensorFlow](https://github.com/tensorflow/tensorflow) project and files in
`onnx/protos` from the [ONNX](https://github.com/onnx/onnx) project; neither
is covered by the licence statement below.

### Apache 2.0 / MIT

All original work is licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any Contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.

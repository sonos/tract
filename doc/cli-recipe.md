# tract cli recipes

`tract` command line is meant to be an auditing, debugging, profiling tool, for CI and
interactive usage.

Please do not make assumptions on the exact forms of its outputs. We do not commit on
any form of stability suitable to script writing.

We are going to use [ONNX mobilenet](../examples/onnx-mobilenet-v2) as examples in these notes. See its
code and README to download a the model.

## Install `tract`

* build latest release: 

```
cargo install tract
```

* download a prebuilt binary fo MacOs Intel, linux Intel, Armv7 or Armv8

```
https://github.com/sonos/tract/releases/latest
```

* run from source

```
git clone https://github.com/sonos/tract
cd cli
cargo run
```

## Model loading

First, `tract` needs to load a model and pipeline it to the preferred `tract-opl` form.
This is equivalent to using the API to load an onnx file, and setting input shapes with
InferenceFact and with_input_fact.

```bash
tract mobilenetv2-7.onnx -i 1,3,224,224
```

loads the model like:

```rust
tract_onnx::onnx()
            .model_for_path("mobilenetv2-7.onnx")?
            .with_input_fact(0, f32.fact(&[1, 3, 224, 224]).into())?
            .into_decluttered()?
```

## Model import pipeline

Once a model is loaded, tract default behaviour is to call the the `dump` subcommand, so the
previous example is equivalent to:

```bash
tract mobilenetv2-7.onnx -i 1,3,224,224,f32 dump
```

The displayed form is `tract-opl` intermediate representation. It is *decluttered* of most
training artefacts, in a form meant to be simple to reason about and as stripped down as
possible.

This is not the "optimised" form: `tract-opl` form is meant to be platform independant, can
be serialized to nnef. The optimised form is just meant to be as fast as possible on a given
CPU.

The `.into_optimized()` transformation can be performed by passing `-O` to the command line.

```bash
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 dump
```

Several other intermediate network "stages" can be reached by using `--pass XXX` instead of `-O`.
`--pass load` and `--pass analyse` are interesting as they can dump a network for which inputs are
unknown (maybe to try and figure out what they could be).

For machine-readable output, pass `--audit-json` to `dump`:

```bash
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 dump --audit-json | jq '.nodes[0]'
```

The default `dump` text output is meant for humans and is awkward to parse;
prefer `--audit-json` from scripts. (Same advice applies to `--profile`,
see below.)

## NNEF round-trip

To convert a loaded network into the tract-OPL (NNEF) form on disk:

```bash
tract mobilenetv2-7.onnx -i 1,3,224,224,f32 dump --nnef model.nnef.tgz
```

To load it back, pass `--nnef-tract-core` (and `--nnef-tract-onnx` if the
network uses ONNX-only extensions) so the parser registers the right
operator set:

```bash
tract --nnef-tract-core model.nnef.tgz dump
```

## Benching a network

We can get a reading of tract performance on a model by running the `bench` or`criterion`
subcommands.

```
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 bench
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 criterion
```

The first one is a simple bench runner customized for tract specific needs, the second one
uses the [criterion](https://docs.rs/criterion) crate.

**`-O` is required for any meaningful number.** Without it, `bench` runs the
decluttered-but-not-optimised graph: generic `Scan` instead of `OptScan`,
`EinSum` / `Conv` instead of lowered `OptMatMul`, no codegen. The result
runs and is numerically correct, but it is several times slower than what
you would actually ship — the dump's op histogram is the tell. The library
equivalent is `model.into_optimized()`; calling `into_runnable()` on a
decluttered model is also valid but measures the same un-optimised graph.
See [`pipeline.md`](pipeline.md) for the full stage breakdown and the
per-runtime variations (`DefaultRuntime` / `MetalRuntime` / `CudaRuntime`).

## Profiling a network

Getting a raw performance number is a first step, but tract can also profile a network execution.
A goto command to get a first glimpse can be:

```
tract -O mobilenetv2-7.onnx -i 1,3,224,224,f32 dump --profile --cost
```

This will show running time for each operator, its relative weight. For some critical operations,
it will also give a number of arithmetic operations per seconds (typically Flops).

Note that
we only count item multiplications, whereas many projects in the HPC field count both
multiplications and additions. So for matrix multiplication, convolution and the like, you may need
to double tract Flops number before comparing with, say, BLAS implementations.

Please do not parse this output. At least use the `--json` output. We do not commit on its stability
but it's less susceptible to changes.

**`--profile` finds hot ops within one graph; it is not a valid A/B between
two graph shapes.** Per-node timing accrues per-node dispatch overhead
(~tens of ns per op on commodity hardware): a path of many small nodes
pays it many times; a single fused op pays it once. Summing per-node
times to compare a fused rewrite against the unfused original
systematically over-credits the fused side. Use `bench` wall-clock for
between-graph comparisons.

## Common timing pitfalls

- **Thermal bias on sustained workloads.** Apple Silicon throttles after
  a few minutes of continuous bench load. Alternating OFF / ON runs
  systematically bias the second half. Batch N OFF runs then N ON runs
  (or insert cooldowns) before trusting a 1-2% delta.
- **WASM benches don't transfer between engines.** Wasmtime/Cranelift
  and V8/TurboFan disagree at the 10-20% level on the same SIMD kernel.
  Measure in the engine you ship to.
- **WASM tier-up.** V8 runs the Liftoff baseline JIT first, then re-JITs
  hot code with TurboFan. First-pass numbers can be 2-4× off steady state;
  warm generously and read steady-state.

## Environment variables

A small set of `TRACT_*` env vars override defaults that are normally fine
out of the box. Most are codegen or CPU-detection knobs you only reach for
when chasing a perf regression or working around an exotic platform; they
apply equally to the library and the CLI.

| Variable | Effect |
|---|---|
| `TRACT_LOG` | `env_logger` filter (e.g. `tract=debug`, `cli=info,tract=warn`). The CLI also derives a default level from `-v` / `-vv`. |
| `TRACT_LAZY_IM2COL_MIN_KERNEL` | Minimum convolution kernel volume before lazy im2col is preferred over eager. Default: 6. Lower it to experiment on memory-constrained targets. |
| `TRACT_LAZY_IM2COL_MAX_EAGER_BYTES` | Scratch-buffer ceiling above which `Conv` switches from eager to lazy im2col. Per-family default (~1 MiB on WASM, ~4 MiB on native). Key knob for the canary-model regression gate. |
| `TRACT_CPU_AARCH64_KIND` | Force aarch64 CPU family detection (`a53`, `a55`, `a72`, `applem`, `generic`, …). Useful for QEMU runs that misreport. |
| `TRACT_CPU_AARCH64_OVERRIDE_CPU_PART` | Force the raw CPU part hex (`0xd03`, …) before the kind-lookup table runs. Lower-level escape hatch when `TRACT_CPU_AARCH64_KIND` doesn't cover the target. |
| `TRACT_CPU_ARM32_NEON` | Force armv7 NEON detection on/off (`true`/`1` or `false`/`0`). |
| `TRACT_CPU_EXPECT_ARM32_NEON` | Used by the test suite to assert the detection result matches what the platform should expose. CI-only. |

The two `LAZY_IM2COL_*` knobs are documented inline next to the constants in
`core/src/ops/cnn/conv/conv.rs`; the CPU-detect knobs in `linalg/src/arm32.rs`
and `linalg/src/arm64.rs`. See [`kernel-notes.md`](kernel-notes.md) for
context on the kernel selection that `LAZY_IM2COL_*` is steering.

## Pulsified networks

The CLI can turn a streaming-friendly network into a pulsified one and run
the assertion path against a batch reference (see also AGENTS.md §Streaming
and pulsification):

```bash
tract --nnef-tract-core model.nnef.tgz --pulse 'T=2' run \
    --input-from-bundle io.npz --assert-output-bundle io.npz
```

The CLI accounts for the accumulated `pulse.delay` when comparing against
the batch reference. Synthetic test cases under `harness/pulse-multi-axis/`
follow this pattern via a `runme.sh` driver.

## Running a test case

`tract` command line can also be use to build test-case, either for non-regression insurance
of debugging purposes.

`--input-facts-from-bundle` takes a `.npz` file, and will set the input facts (dtype, shape) according to the tensors
in the npz file. This is useful when your model does not have any input type information embedded within it.

The `run` subcommand accepts an `--input-from-bundle` that also takes a `.npz` file, but it
will not set any input fact, it will only take the tensor values.
This will also supersede the `-i` option: we will take the input shapes and tensor
from the input itself.

The `run` subcommand also accepts an `--assert-output-bundle`. This time, the tensors names are
matched with the model output names. `tract` will run over the input and check that its finding
are the same to the expected output (with some leeway for rounding differences).

[Example here](/onnx/test_cases/qtanh_1) for a quantized tanh in onnx.

```sh
tract model.onnx -O run --input-from-bundle io.npz --assert-output-bundle io.npz
```

If we want to make sure we actually check something, `-v` can help:

```
tract -v model.onnx -O run --input-from-bundle io.npz --assert-output-bundle io.npz
```

The log displays "Checked output #0, ok." (among other information).

[generate_io.py here](/onnx/test_cases/transformer-mlm/generate_io.py) contains an example building a
testcase for a BERT model from huggingface for inspiration.

## Saving outputs

The `--save-outputs` (long form `--save-outputs-npz`) flag on the `run` subcommand writes the
model outputs to an `.npz` file after execution. This is the easiest way to capture a reference
output for a given input, which can then be used with `--assert-output-bundle` in a later run.

```sh
# capture outputs from a first run
tract -O model.onnx run --input-from-bundle inputs.npz --save-outputs reference.npz

# replay and assert on a subsequent run (e.g. after a code change)
tract -O model.onnx run --input-from-bundle inputs.npz --assert-output-bundle reference.npz
```

Output tensors are keyed by their model output name (or `output_N` if unnamed).

There is also `--save-outputs-nnef` which writes each output tensor as a separate `.dat` file
in a folder, in NNEF layout — useful for inspecting individual tensors with external tools.

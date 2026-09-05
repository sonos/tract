# tract — codebase overview

tract is Sonos' neural-network inference engine written in Rust.
It reads ONNX, NNEF, TensorFlow Lite, and TensorFlow models, optimises them,
and runs them on CPU (x86/ARM), GPU (Metal, CUDA), embedded targets, and WASM.

This is the operational quick reference: where things live, how to build and
test, and the abstractions you need to work with the graph. It is descriptive —
the rules you must follow when contributing are in
[`AGENTS.md`](../AGENTS.md) (also readable as `CLAUDE.md`), and take precedence
over anything here.

For conceptual background not derivable from the source, see the rest of
this directory — [`README.md`](README.md) indexes it.

---

## Crate map

| Crate | Purpose | Depends on |
|---|---|---|
| `data` | `Tensor`, `DType`, `TractResult`, low-level storage | _(none)_ |
| `linalg` | Micro-kernel dispatch (BLAS-style, hand-rolled SIMD) | `data` |
| `core` | `TypedModel`, op trait, passes, rewriter, `TypedModelPatch` | `linalg`, `data` |
| `hir` | Untyped inference graph (pre-type-analysis) | `core` |
| `nnef` | NNEF load/save, tract-OPL extensions, NNEF ser/de for core ops | `core` |
| `onnx-opl` | tract-OPL extensions for ONNX-specific ops | `nnef`, `extra` |
| `onnx` | ONNX importer | `nnef`, `hir`, `onnx-opl`, `extra` |
| `tflite` | TensorFlow Lite importer | `core` |
| `tensorflow` | TensorFlow importer | `hir`, `pulse` |
| `pulse-opl` | Streaming op primitives | `nnef` |
| `pulse` | Streaming / causal inference | `pulse-opl`, `transformers` |
| `transformers` | Transformer-specific ops (RmsNorm, Silu, GeluApproximate, ...) | `nnef`¹ |
| `gpu` | Shared GPU abstractions | `core`, `pulse-opl`, `transformers` |
| `metal` | Apple Metal backend | `gpu`, `core`, `pulse-opl`, `transformers` |
| `cuda` | NVIDIA CUDA backend | `gpu`, `core`, `pulse-opl`, `transformers` |
| `extra` | Miscellaneous ops not yet in core | `nnef`, `pulse` |
| `cli` / `libcli` | `tract` command-line tool | most of the above |
| `api/rs` | High-level stable public Rust API | `nnef`, `onnx`, `pulse`, `transformers`, `metal`, `cuda`, ... |
| `api/ffi` | C FFI over `api/rs` | `api/rs` |

¹ `transformers` has no direct `tract-core` dep; import core types via `tract_nnef::tract_core`.

---

## Build and test

```sh
# build everything
cargo build --workspace

# test a single crate
cargo test -p tract-core

# test the whole workspace
cargo test --workspace

# format (always repo-wide, never per-crate; rust-toolchain.toml pins stable, so bare cargo fmt is correct)
cargo fmt --all

# lint
cargo clippy --workspace
```

The `harness/` directory contains integration tests that run against real
models. `.travis/native.sh` runs the full native Linux CI suite; running it
locally requires `libssl-dev` (needed by the `tflite` step).

Synthetic NNEF tests under `harness/nnef-test-cases/`
are driven by a `runme.sh` that calls the `tract` CLI with `--assert-output-bundle`
against a reference `io.npz`. Add new cases there rather than as Rust integration
tests. If the assertion you need isn't expressible through the CLI, extend the CLI.

### Model inspection

```sh
# human-readable graph dump
tract model.nnef.tgz dump

# machine-readable — pipe to jq or python
tract model.nnef.tgz dump --audit-json | jq '.nodes[] | select(.op_name == "Conv")'
```

Reach for `--audit-json` when scripting; the plain `dump` output is meant for
humans and is awkward to parse.

### Hardware and kernel benchmarking

`tract hwbench <M,K,N[,dt]>...` reports the machine (cores, cache/memory
bandwidth) and times every matmul micro-kernel at each shape, marking the
dispatcher's pick with `<--` (`--json` to parse, `--assert` to gate on it) — the
tool for diagnosing or calibrating kernel selection. See
[`doc/kernel-notes.md`](kernel-notes.md) and
[`doc/cli-recipe.md`](cli-recipe.md#hardware-and-kernel-benchmarking).

### test-rt

`test-rt` is the cross-backend test framework. It separates test suites from
runtimes: a suite (e.g. `suite-unit`, `suite-onnx`) defines a set of
`Test`-trait objects; a runner crate (e.g. `test-unit-core`, `test-metal`,
`test-cuda`) picks a `Runtime` implementation and runs a subset of those
suites against it, with its own ignore list.

Layout:

| Crate | Role |
|---|---|
| `test-rt/infra` | `Test`, `TestSuite`, `Runtime` traits and test-runner harness |
| `test-rt/suite-unit` | Unit tests for core ops (conv, einsum, matmul, ...) |
| `test-rt/suite-onnx` | ONNX backend test suite |
| `test-rt/test-unit-core` | Runs `suite-unit` on the default CPU runtime |
| `test-rt/test-onnx-core` | Runs `suite-onnx` on the default CPU runtime |
| `test-rt/test-metal` | Runs unit + onnx suites on the Metal backend |
| `test-rt/test-cuda` | Runs unit + onnx suites on the CUDA backend |
| `test-rt/test-f16` | Runs f16-specific cases |
| `test-rt/test-tflite` | Runs suite against the TFLite runtime |
| `test-rt/test-nnef-cycle` | Verifies NNEF round-trip for all suite cases |

To add a new op test: add a case to the relevant `suite-*` crate. The runner
crates pick it up automatically; add an ignore entry in the runner only if the
backend genuinely cannot support the case.

---

## Core abstractions

> **Client code** (applications, examples, language bindings) should use `api/rs`
> only. The internal crates (`core`, `nnef`, `onnx`, ...) are not stable API surface.
> When asked "is X part of the public API?", check `api/rs/src/lib.rs` — that is
> the authoritative surface, not the internal crate's `pub` items.

### key principle

tract avoid specializing for one model or an application. Model-wide behaviours or
optimisations should emerge from op-scoped manipulation composing together. There are
pragmatic compromises: sometimes introducing "big" primitives than could be implemented 
with atomic operators (Convolution, Attention, RmsNorm, ...) is unlocking optimisations.

### TypedModel / TypedNode / TypedFact

The main IR. A `TypedModel` is a DAG of `TypedNode`s. Each node holds a boxed
`Op` and a list of `TypedFact` outputs (element type + symbolic shape).

### Op trait (`core/src/ops/mod.rs`)

Every op implements `Op` and usually `TypedOp`. Key methods:

- `eval` -- eager execution
- `output_facts` -- shape/type inference
- `declutter` -- return a `TypedModelPatch` to simplify the op or replace it
- `codegen` -- return a patch targeting a specific backend/platform

## Model rewriting

`TypedModelPatch` is the **preferred** way to modify a model. Direct mutation of
`TypedModel` nodes or edges is an exception reserved for construction and
well-understood bulk transforms -- in almost every other case, implement rule
functions that return `TypedModelPatch`es, and wrap them in a `Rewriter` to
define a `ModelTransform`.

### TypedModelPatch (`core/src/model/patch.rs`)

Surgical edits to a model. Build a patch, then `patch.apply(model)`.

### Rewriter (`core/src/model/rewriter.rs`)

Per-op declutter rules collected into a typed dispatch table:

```rust
Rewriter::default()
    .with_rule_for::<MyOp>("rule-name", |ctx, model, node, name, op| {
        rule_if!(some_condition);
        rule_if_some!(x = maybe_value);
        // build and return a TypedModelPatch
        Ok(Some(patch))
    })
```

Use `Rewriter` for rules that fire on a single op type. The three guard macros
(defined in `core/src/transform.rs`) provide early-return ergonomics inside a
`TractResult<Option<TypedModelPatch>>` body:

| Macro | Exits when |
|---|---|
| `rule_if!(cond)` | `cond` is false |
| `rule_if_let!(pat = expr)` | pattern does not match |
| `rule_if_some!(pat = expr)` | value is `None` |

### ModelTransform (`core/src/transform.rs`)

Whole-model passes (e.g. float precision translation, block-quant folding).
Implement the `ModelTransform` trait when you need to walk the entire graph
rather than react to individual op types.

### When to use which

| Situation | Tool |
|---|---|
| Simplify / fuse one op type | `Op::declutter` + `TypedModelPatch` |
| Cross-op pattern (N ops -> M ops) | `Rewriter` rule |
| Whole-model structural change | `ModelTransform` |
| Backend lowering for one op | `Op::codegen` + `TypedModelPatch` |

---

## Op detection via declutter

Tract detects and fuses transformer ops during the declutter pass, not in a
separate recognition pass:

- `RmsNorm` -- detected in `Reduce::declutter`
- `Silu` -- detected in `Sigmoid::declutter` (via `element_wise!` `; declutter:` param)
- `GeluApproximate` -- detected in `Pow::declutter` (chained `declutter_pow`)

---

## Streaming and pulsification

Streaming inference converts a `TypedModel` into a `PulsedModel`
(`pulse/src/model.rs`) that processes a fixed-size chunk along one axis at each
step.

- **Streaming axis and pulse size.** Each op pulsifies through an impl in
  `pulse/src/ops/`. The streaming axis is tracked node-by-node; an op that
  reorders or merges axes must declare what the streaming axis becomes on its
  output, otherwise pulsification fails or produces wrong delays.
- **Delay.** Ops that need past context (conv, attention windows, banded masks)
  insert a `Delay` op from `pulse-opl/src/delay.rs` to buffer earlier pulses.
  The accumulated output delay is exposed as `pulse.delay` and used by the CLI
  assertion path to skip warmup before comparing against a batch reference.
- **ChangeAxes.** `core/src/optim/change_axes.rs` rewrites axis layout during
  optimisation. Any change-axes interaction has to preserve the streaming axis
  identity, so new ops that move axes around need a `change_axes` impl that
  agrees with their pulsification.

The streaming model has subtle invariants -- don't touch `pulse` / `pulse-opl`
casually.

---

## NNEF serialisation

NNEF ser/de for core ops lives in `nnef/src/ops/core/`. Ops are registered
with a primary `tract_core_*` name and backward-compatible `tract_transformers_*`
aliases where needed.

Re-export shims in `transformers/src/ops/mod.rs` keep downstream crates
(`cuda`, `metal`, `gpu`, `test-rt`) working without a direct `tract_core` dep.

---

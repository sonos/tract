# tract — Agent Guide

tract is Sonos' neural-network inference engine written in Rust.
It reads ONNX, NNEF, TensorFlow Lite, and TensorFlow models, optimises them,
and runs them on CPU (x86/ARM), GPU (Metal, CUDA), embedded targets, and WASM.

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
| `transformers` | Transformer-specific ops (RmsNorm, Silu, GeluApproximate, …) | `nnef` |
| `gpu` | Shared GPU abstractions | `core`, `pulse-opl`, `transformers` |
| `metal` | Apple Metal backend | `gpu`, `core`, `pulse-opl`, `transformers` |
| `cuda` | NVIDIA CUDA backend | `gpu`, `core`, `pulse-opl`, `transformers` |
| `extra` | Miscellaneous ops not yet in core | `nnef`, `pulse` |
| `cli` / `libcli` | `tract` command-line tool | most of the above |
| `api/rs` | High-level stable public Rust API | `nnef`, `onnx`, `pulse`, `transformers`, `metal`, `cuda`, … |
| `api/ffi` | C FFI over `api/rs` | `api/rs` |

**Important:** `transformers` does **not** directly depend on `tract-core`.
It accesses core types via `tract_nnef::tract_core`. Use that path in imports,
not `tract_core::…` directly.

---

## Build and test

```sh
# build everything
cargo build --workspace

# test a single crate
cargo test -p tract-core

# test the whole workspace
cargo test --workspace

# format (always repo-wide, never per-crate; pin the toolchain to avoid spurious diffs)
cargo +1.91.0 fmt --all

# lint
cargo clippy --workspace
```

The NNEF round-trip test is the gold standard for op correctness: if an op
serialises to NNEF and deserialises back to an identical graph it is correct by
construction.

The `harness/` and `test-rt/` directories contain integration tests that run
against real models. `.travis/native.sh` runs the full native Linux CI suite;
running it locally requires `libssl-dev` (needed by the `tflite` step).

---

## Core abstractions

> **Client code** (applications, examples, language bindings) should use `api/rs`
> only. The internal crates (`core`, `nnef`, `onnx`, …) are not stable API surface.

### TypedModel / TypedNode / TypedFact

The main IR. A `TypedModel` is a DAG of `TypedNode`s. Each node holds a boxed
`Op` and a list of `TypedFact` outputs (element type + symbolic shape).

### Op trait (`core/src/ops/mod.rs`)

Every op implements `Op` and usually `TypedOp`. Key methods:

- `eval` — eager execution
- `output_facts` — shape/type inference
- `declutter` — return a `TypedModelPatch` to simplify the op or replace it
- `codegen` — return a patch targeting a specific backend/platform

## Model rewriting

`TypedModelPatch` is the **preferred** way to modify a model. Direct mutation of
`TypedModel` nodes or edges is an exception reserved for construction and
well-understood bulk transforms — in almost every other case, implement rule
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
| Cross-op pattern (N ops → M ops) | `Rewriter` rule |
| Whole-model structural change | `ModelTransform` |
| Backend lowering for one op | `Op::codegen` + `TypedModelPatch` |

---

## Op detection via declutter

Tract detects and fuses transformer ops during the declutter pass, not in a
separate recognition pass:

- `RmsNorm` — detected in `Reduce::declutter`
- `Silu` — detected in `Sigmoid::declutter` (via `element_wise!` `; declutter:` param)
- `GeluApproximate` — detected in `Pow::declutter` (chained `declutter_pow`)

---

## NNEF serialisation

NNEF ser/de for core ops lives in `nnef/src/ops/core/`. Ops are registered
with a primary `tract_core_*` name and backward-compatible `tract_transformers_*`
aliases where needed.

Re-export shims in `transformers/src/ops/mod.rs` keep downstream crates
(`cuda`, `metal`, `gpu`, `test-rt`) working without a direct `tract_core` dep.

---

## Branch and commit hygiene

```sh
# always branch off origin/main without setting upstream
git checkout -b my-feature origin/main --no-track
```

- Run `cargo fmt --all` before every commit. Metal source files need formatting
  too, even when building on Linux.
- Do not add "Co-Authored-By" or similar trailer lines to commit messages.
- Do not push or fetch; the human handles all remote operations.
- Commit at natural checkpoints during large multi-crate refactors — commits
  are cheap and make bisection easy.

---

## Style

  Commit messages:
  - State what was wrong and the fix — no consequence chains ("X broke Y broke Z").
  - One short paragraph; skip "Result:/Consequence:/Symptom:" sections and laundry-lists of every place the bug surfaced.

  Inline code comments:
  - Default to none. Code MUST be self-explanatory via variable and function naming.
  - In tract, an inline comment is a signal that something implicit is happening — hidden constraint, non-obvious invariant, bug workaround.
  - (This has not been enforced consistently since codebots became a thing.)
  - Avoid section banners (// ── Step 2: Pad → Reshape ──), prefer split in functions. It's ok to have long function prototype in private function (within reason) #[allow(clippy::too_many_arguments)] authorized in such case.

  Formatting:
  - Always run `cargo +1.91.0 fmt --all` before committing — bare `cargo fmt` uses a newer rustfmt and produces spurious diffs CI rejects.

  PR comments and review replies:
  - Open PR with a short, crystal-clear summary paragraph — one or two sentences stating what the PR is about and why it matters, before details if necessary.
  - Follow-up questions and comments on a PR must be handled by humans only, bots and LLM are forbidden after the PR opening message post.

## Things to avoid

- **Clap extension traits** — use the clap API directly, even with turbofish.
- **Mocking internals in tests** — prefer real model round-trips.
- **Hand-rolling model-walk loops** — reach for `Rewriter`, `ModelTransform`,
  or `TypedModelPatch` instead.
- **Adding abstraction beyond the task** — three similar lines beat a premature
  helper.
- **Touching `pulse`/`pulse-opl` without understanding causal inference** —
  the streaming model has subtle invariants around axis tracking and delay.

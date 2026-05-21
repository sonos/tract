# Load → optimise → run

This is the canonical pipeline a model goes through between disk and a
`Runnable` you can call `.run()` on. Each stage is a separate method, so
you can stop at any point to inspect or serialise the intermediate.

## Public API

The pipeline shows up in the public API as a chain of typed wrappers, one
per stage. The happy path on CPU is two method calls between loading and
running:

```rust
use tract::prelude::*;

let model = tract::onnx()?
    .load("model.onnx")?            // InferenceModel
    .into_model()?                  // Model
    .into_runnable()?;              // Runnable

let outputs = model.run(tvec!(input.into()))?;
```

What each call buys you:

- `.load(path)` — opens a framework file and returns an `InferenceModel`,
  the partial-shape/type form. Bind partial shapes here with
  `set_input_fact(ix, "1,3,224,224,f32")`.
- `.into_model()` — resolves all shapes and types and hands back a
  `Model` in tract-OPL form. This is the portable, NNEF-serialisable
  representation; you can `.transform("f32_to_f16")` it or write it to
  disk before going any further.
- `.into_runnable()` — produces a `Runnable` via the **`default`**
  runtime, which is the in-process CPU implementation. Codegen and
  kernel selection happen here; if you skipped this call and serialised
  the `Model` instead, you would have something that loads on any
  machine.

Loading from NNEF is one step shorter: `tract::nnef()?.load(path)?` hands
back a `Model` directly. NNEF already carries fully-resolved shapes and
types, so there is nothing for the `InferenceModel` stage to resolve.
This is the recommended deployment shape — translate to NNEF once (with
the CLI or once at startup), ship that, and skip the framework loaders
at runtime.

For a non-CPU runtime, pick one explicitly and prepare against it:

```rust
let rt = tract::runtime_for_name("metal")?;     // or "cuda", "default", ...
let runnable = rt.prepare(model)?;
```

`runtime_for_name` looks up the runtime by name in the `inventory`-based
registry; whichever runtime crates are linked into the binary contribute
to the pool (`tract-metal`, `tract-cuda`, ...). `.into_runnable()` is
exactly the shortcut for `runtime_for_name("default")?.prepare(model)?`
— same code path, the CPU runtime just happens to be registered under
that name.

## CLI and internals

This is the *what is `.into_runnable()` actually doing* view: the
pipeline stage by stage, the `Runtime` trait that owns the second half,
and how the CLI's flags surface each piece.

### The stages

| Stage | Method | Lives in | Output |
|---|---|---|---|
| Load | `tract::onnx().load(path)` / `tract::nnef().model_for_path(path)` | `tract-onnx` / `tract-nnef` | `InferenceModel` |
| Analyse | `InferenceModel::into_typed()` | `tract-hir` | `TypedModel` (full shapes/types) |
| Declutter | `TypedModel::into_decluttered()` | `tract-core` | portable, NNEF-serialisable tract-OPL |
| Optimise | `TypedModel::optimize()` | `tract-core` | target-specific LIR (codegen, kernel selection) |
| Plan | `TypedSimplePlan::new_with_options(model, options)` | `tract-core` | `Runnable` |
| Spawn / run | `Runnable::spawn() -> State` ; `State::run(inputs)` | `tract-core` | tensors |

Two convenience wrappers stitch the middle stages:

- `TypedModel::into_decluttered()` = declutter only.
- `TypedModel::into_optimized()` = declutter + optimise.

### Decluttered vs optimised

**Decluttered** is the portable form: training artefacts are gone, obvious
patterns are fused (`RmsNorm`, `Silu`, ...), but the operator set is still
the high-level tract-OPL one — `EinSum`, `Conv`, generic `Scan`, etc. This
is what you serialise to NNEF, what `tract dump` shows by default, and
what crosses a CPU/GPU split unchanged.

**Optimised** is target-specific: `EinSum`/`Conv` get lowered to
`OptMatMul` over the platform's micro-kernels (`avx512_mmm_f32_*`,
`arm64simd_mmm_f32_*`, ...), `Scan` becomes `OptScan`, and per-machine
codegen patches are applied. This is what runs; you don't serialise it
because it's only valid for the machine that produced it.

Two consequences worth knowing:

- **Optimisation isn't optional for timing.** Running a decluttered model
  is numerically correct but several times slower than what you'd ship
  (see [`cli-recipe.md`](cli-recipe.md) § Benching for the `-O` callout).
- **"Optimised" means different things to different runtimes.** The CPU
  runtime's `into_optimized()` differs from what Metal or CUDA does
  before they hand off to their own kernels — see the next section.

### The Runtime trait

`tract_core::runtime::Runtime` owns the "from typed model to runnable"
half of the pipeline. A runtime's `prepare(model: TypedModel)` method
applies whatever transform / codegen its target needs, then wraps the
result in a `TypedSimplePlan`:

```rust
pub trait Runtime: Debug + Send + Sync + 'static {
    fn name(&self) -> StaticName;
    fn prepare(&self, model: TypedModel) -> TractResult<Box<dyn Runnable>>;
    /* ... */
}
```

Concrete runtimes shipped in-tree:

| Runtime | Crate | What `prepare` does |
|---|---|---|
| `DefaultRuntime` | `tract-core` | `model.into_optimized()` then `SimplePlan` |
| `MetalRuntime` | `tract-metal` | `MetalTransform::default()` then `into_optimized()` then `SimplePlan` |
| `CudaRuntime` | `tract-cuda` | `CudaTransform` then `optimize()` then `SimplePlan` |
| `UnoptimizedRuntime` | `tract-cli` (CLI only) | `SimplePlan` straight over the typed model — no optimise |

A runtime registers itself via `register_runtime!` (an `inventory`-based
registry), so any crate linked into the binary contributes to the runtime
pool. The CLI's `--runtime` flag and the library's `runtime_for_name(s)`
lookup go through that registry.

Single-threaded ARM CPU loads are the primary production target; x86_64,
Apple Silicon (CPU and Metal), CUDA, and WASM are all supported.

`UnoptimizedRuntime` only exists for the CLI — it lets `tract --pass …
run` execute the intermediate form (decluttered, before-optimise, etc.)
for inspection or `--assert-output-bundle` round-trips. It is not what
you want for performance numbers; mistaking it for `DefaultRuntime` is
the source of the "tract is silently slow" trap.

### How the CLI maps to runtimes

- **`tract … run`** (no `-O`) — runs `UnoptimizedRuntime` over whatever
  stage `--pass` last produced (default: `before-optimize`). Correct,
  slow.
- **`tract -O … run`** — runs `DefaultRuntime`, which calls
  `into_optimized()`. Production path; same as the library's
  `.into_runnable()`.
- **`tract -O --metal … run`** / **`--cuda …`** / **`--runtime <name>`**
  — picks a specific runtime by name. Each runtime applies its own
  pre-plan transform (e.g. `MetalTransform` inserts Metal-side dispatch
  ops, then standard `into_optimized` lowers what's left of the CPU
  part).

## Example rewrites

Two distinct pipeline stages rewrite the graph: **declutter** is
target-independent (same regardless of which `Runtime` is going to
run the result) and **lowering** is the per-target `codegen` step that
turns high-level ops into the primitives a specific machine actually
runs. The subsections below give a small, non-exhaustive sampling of
each; the source of truth is the `*::declutter` and `*::codegen`
methods on the relevant ops.

### Declutter

Decluttering goes in two directions at once: it **decomposes** some
high-level ops into primitives, and it **fuses** recognisable chains
of primitives back into one high-level op. The op set after declutter
is different from the framework's source op set in both directions —
worth knowing when reading a `dump`, because *"my model had N
`LayerNorm` and I see zero of them"* is not a bug.

Which way a given op goes is a pragmatic call driven by optimisation
opportunity and operator prevalence. A pattern is fused when it is
common enough to be worth its own kernel (`RmsNorm`, `Silu`,
`GeluApproximate` — high prevalence in transformers, concentrated
payoff from a dedicated implementation). A high-level op is decomposed
when the smaller pieces compose better with neighbouring ops or expose
optimisations the wrapping op would have hidden. The same
`LayerNormalization` runs both ways: the inner RmsNorm gets its
dedicated kernel, and the surrounding mean-subtract and γ/β affine are
left as primitives that downstream rewrites can pick up.

Decompositions (one upstream op → several primitives):

- `MatMul` and `Gemm` → `EinSum`. tract has no first-class matmul op;
  every framework-side matrix-multiply lands as an `EinSum` with an
  axes spec (carried since `to_typed`, so by the time you see the
  typed graph there is no `MatMul` node left).
- `LayerNormalization` → `Sub(mean)` + `RmsNorm` (the inner
  `rsqrt(mean(x²) + ε)` chain) + `Mul(γ)` + `Add(β)`. A transformer
  with N `LayerNorm` shows *0 `LayerNorm` + N `RmsNorm`* in the dump,
  with the surrounding `Sub` / `Mul` / `Add` ops still present — that
  histogram is correct.
- `AveragePool` → `SumPool(normalize = true)`.
- `HardSigmoid` → a `Clip` / `Min` / `Max` chain.
- `Resize` (nearest, integer scales) → `Reshape` → `AddAxis` →
  `MultiBroadcastTo` → `Reshape` tile chain.
- `Conv` on unit-batch input → the batch dim is peeled, inner convs
  end up rank-3 `CHW` rather than rank-4 `NCHW`.

Fusions (a primitive chain → one high-level op):

- `Mul(rsqrt(reduce_mean(x²) + ε))` → `RmsNorm` — fires in
  `Reduce::declutter`.
- `x · sigmoid(x)` → `Silu` — fires in `Sigmoid::declutter`.
- The GELU-approximate polynomial → `GeluApproximate` — chained
  `declutter_pow` in `Pow::declutter`.

### Lowering

These run during `optimize()` and are per-target: the active `Runtime`
decides what fires. The examples below are what `DefaultRuntime`
(CPU) produces. Not material you usually audit from `dump`, but
useful context when reading a profile or chasing a perf surprise.

- `EinSum` lowers to one or more `OptMatMul`s once axis identities
  resolve to concrete `M, K, N` patterns; the same `EinSum` op can
  surface as several different shape signatures depending on the
  surrounding declutter. This is the path every framework-side
  matmul-shaped op (`MatMul`, `Gemm`, the linear inside `Conv`'s
  im2col branches) eventually goes through.
- `Conv` codegen (`core/src/ops/cnn/conv/conv.rs::codegen`) tries
  four lowerings in order:
  1. quantised im2col + matmul, if the op has `q_params`;
  2. lazy im2col + matmul, if the input shape is concrete and the
     kernel volume / scratch ceiling favour it (see
     `TRACT_LAZY_IM2COL_*`);
  3. direct depthwise, if `group != 1 && group == in_channels == out_channels`;
  4. eager im2col + matmul otherwise.
  Branches 1, 2, 4 all produce an `OptMatMul` over an im2col buffer;
  branch 3 is the only non-im2col convolution.
- `Scan` → `OptScan` keeps a persistent inner-body plan
  (`model_state: TypedSimpleState`) across iterations, but each
  iteration is a full `model_state.run(inputs)`: `set_inputs` →
  `resolve_symbols_with_states` → `exec_plan` → `outputs` →
  `reset_turn`. `reset_turn` clears `resolved_symbols`, so any
  loop-invariant symbol is re-resolved on every step. Worth knowing
  when profiling tight RNN / decoder loops.

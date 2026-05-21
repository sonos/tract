# Tract

tract is a neural network inference library. It takes trained networks from
higher-level frameworks (TensorFlow, PyTorch, etc.), converts them to an
intermediate representation, and runs them on end-user data. It is designed
to be portable and embedding-friendly, with a focus on inference on the
edge, in the browser, or on small embeddable CPUs.

For an overview of the codebase (crates, traits, model rewriting, streaming,
CLI inspection) see [`AGENTS.md`](../AGENTS.md). The notes in this directory
cover material that is harder to derive from reading the source:

* this file — the tract-OPL philosophy and the translate-time / runtime split
* [`pipeline.md`](pipeline.md) — load → optimise → run, and the `Runtime` trait
* [`symbolic-shapes.md`](symbolic-shapes.md) — `TDim`, `Symbol`, and how to bind them
* [`graph.md`](graph.md) — Graph, Node, Fact, Op concepts
* [`op.md`](op.md) — anatomy of an Op (Op / EvalOp / TypedOp / InferenceOp)
* [`cli-recipe.md`](cli-recipe.md) — `tract` command-line cookbook
* [`kernel-notes.md`](kernel-notes.md) — tract-linalg kernels and debugging
* [`nnef/`](nnef/) — reference schemas for tract NNEF extensions

## Public API

Client code (applications, examples, language bindings) should use the
`api/rs` crate. The authoritative surface is `api/rs/src/lib.rs`. The
internal crates (`core`, `nnef`, `onnx`, ...) are not stable API.

## tract-OPL

Tract OPL (Operation Programming Language) is an NNEF-based intermediate
representation of a neural network. NNEF aims to be for *inference* what
ONNX is for *training* frameworks — inference engines and training
frameworks have widely divergent requirements, so OPL keeps the operator
surface narrow and machine-independent.

We extend NNEF with fragments for tract-core operators that NNEF does not
cover and for ONNX/TF operators we chose to keep out of tract-core. This
lets us split the operator surface across crates and shrink the runtime
footprint:

* The `tract` command line includes `tract-onnx`. It can translate an ONNX
  network to a tract-core-plus-extensions model in memory and dump it as
  NNEF. This is normally done once, right after training.
* At runtime, you ship only `tract-core`, `tract-nnef` (the parser), and
  optionally `tract-onnx-opl` for the handful of ONNX-only operators.

The same translate-time / runtime split applies to streaming: `tract-pulse`
turns a regular model into a pulsified one (via the command line if needed),
and only the much smaller `tract-pulse-opl` is needed at runtime.

The tract-OPL format is machine-independent. Calling `into_optimized()` on a
loaded NNEF network produces the most efficient form for the current
machine; this is much cheaper than the full decluttering pass that runs
when loading directly from a training format. For the full pipeline
mechanics — what `into_optimized` actually runs, how the `Runtime` trait
fits in, and what `MetalRuntime`/`CudaRuntime` do differently — see
[`pipeline.md`](pipeline.md).

The NNEF extensions are documented as reference schemas in
[`nnef/`](nnef/) — `tract_core`, `tract_onnx`, `tract_pulse`, and
`tract_resource`.

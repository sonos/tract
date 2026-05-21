# tract internals documentation

Internal notes about tract. Start from [`AGENTS.md`](../AGENTS.md) for the
operational quick reference (crate map, build/test, model rewriting,
streaming, CLI inspection); the documents here cover conceptual material
that is harder to derive from reading the source.

* [`intro.md`](intro.md) — what tract is and the tract-OPL design
* [`pipeline.md`](pipeline.md) — load → optimise → run, and the `Runtime` trait
* [`symbolic-shapes.md`](symbolic-shapes.md) — `TDim`, `Symbol`, and how to bind them
* [`graph.md`](graph.md) — Graph, Node, Outlet, Fact, model pipeline
* [`op.md`](op.md) — anatomy of an Op (`Op` / `EvalOp` / `TypedOp` / `InferenceOp`)
* [`cli-recipe.md`](cli-recipe.md) — `tract` command-line cookbook
* [`kernel-notes.md`](kernel-notes.md) — tract-linalg kernels and debugging
* [`nnef/`](nnef/) — reference schemas for the `tract_*` NNEF extensions

Documentation drifts faster than code. If something here disagrees with the
source, trust the source — and consider patching the doc.

# Design sketch: a layer-range partitioner for tract-core

## Motivation / gap
tract has **no dedicated way to slice a `TypedModel` into standalone sub-models**.
Everything needed exists as primitives — `set_input_outlets`, `select_output_outlets`
(`core/src/model/graph.rs`), `IntoTranslator::translate_model`
(`core/src/model/translator.rs`), `compact` — but a user must compose them by hand and,
critically, there is **no validation**: if the chosen boundary doesn't dominate the region,
`translate_model` silently pulls in the original `Source`s and yields an under-specified or
wrong sub-model. This bites pipeline/tensor parallelism, partial evaluation, test-fixture
extraction, and per-subsystem debugging alike.

Proposal: one small, general, **model-agnostic** primitive plus a thin pipeline convenience,
in `tract-core`. No new ops. Everything model-specific (block detection, KV grouping,
transport) stays out of core.

## Proposed API (tract-core, `core/src/model/`)

```rust
/// Extract the sub-graph of `model` computing `outputs`, with `outputs`' cone
/// bounded by `inputs`: each interior outlet in `inputs` becomes a fresh
/// `Source` carrying that outlet's fact; nodes not on an inputs→outputs path are
/// pruned; consumed consts are copied in; the `SymbolScope` is carried over.
///
/// Errors (this is the value over raw primitives):
/// - `UndeclaredBoundaryInput { outlet }` — the outputs' cone reaches a `Source`
///   (or nullary non-const) that is NOT in `inputs`: a boundary tensor the caller
///   forgot to declare. (This is the mask / positional-encoding / KV `Source→Concat`
///   foot-gun turned into a clear error.)
/// - `StatefulBoundary { node }` — an outlet in `inputs`/`outputs` is the output
///   of a stateful op (KV-cache, `Scan`); its `OpState` can't be reconstructed on
///   the far side of a cut, so it may not be a boundary.
///
/// Adds no ops; the result is a standalone runnable `TypedModel`.
pub fn extract_subgraph(
    model: &TypedModel,
    inputs: &[OutletId],
    outputs: &[OutletId],
) -> TractResult<TypedModel>;

/// Split into a linear pipeline of `boundaries.len() + 1` stages. `boundaries[i]`
/// is the set of outlets crossing the cut between stage `i` and stage `i+1`
/// (usually one residual outlet). Convenience over `extract_subgraph`:
///   stage 0     : inputs = model.inputs,   outputs = boundaries[0]
///   stage i     : inputs = boundaries[i-1], outputs = boundaries[i]
///   stage last  : inputs = boundaries[n-1], outputs = model.outputs
///
/// Errors additionally with `CrossStageFanout { outlet, from, to }` when a cut
/// outlet is consumed beyond the immediately-following stage (a skip connection
/// spanning >1 stage) — pure pipelining can't express that without promoting the
/// outlet as an input to every downstream stage that needs it. Decoder-only LMs
/// don't hit this (only the residual crosses to the next block); the check makes
/// the limitation explicit rather than silently wrong.
pub fn split_pipeline(
    model: &TypedModel,
    boundaries: &[Vec<OutletId>],
) -> TractResult<Vec<TypedModel>>;
```

Ergonomic sugar (optional): `TypedModel::subgraph(&self, inputs, outputs)` /
`TypedModel::split_pipeline(&self, boundaries)` forwarding to the free functions.

## Implementation (thin; the novelty is validation)
`extract_subgraph`:
1. **Validate** with `eval_order_for_nodes(model.nodes(), inputs, outputs, &[])`
   (`core/src/model/order.rs`): walk the outputs' cone; if it reaches any `Source`
   not in `inputs`, return `UndeclaredBoundaryInput`. Check each `inputs`/`outputs`
   producer's `op.is_stateless()`; else `StatefulBoundary`.
2. `let mut m = model.clone(); m.set_input_outlets(inputs)?; m.select_output_outlets(outputs)?;`
3. `let mut sub = IntoTranslator.translate_model(&m)?; sub.compact()?;` — prunes to the
   cone, turns boundary nodes into `Source`s, copies consumed consts, renumbers ids.
4. Return `sub`. (Preserves symbols; per-stage symbol *resolution* stays the caller's
   responsibility — feed consistent boundary shapes.)

`split_pipeline` is a fold of `extract_subgraph` over consecutive boundary pairs, plus the
`CrossStageFanout` check via `outlet_successors` (`graph.rs:601`).

Both are pure functions (1 model → N models), so they are **not** `ModelTransform`s (which are
1→1 in place). They use `translate_model` — the sanctioned framework — not a hand-rolled walk,
matching `CLAUDE.md` ("use TypedModelPatch/Rewriter/ModelTransform, don't hand-roll").

## Explicitly OUT of core (stays in an adapter crate / example)
- **Block-boundary discovery** — mapping a "layer index" to a cut `OutletId`. tract imports an
  opaque graph, so this is name-heuristic (`..._<N>_xAdd2_1`) or NNEF-metadata driven and is
  inherently model/exporter-specific. Keep it in `tract-distributed` (or wherever), not core.
- **Stateful-I/O role grouping** — classifying model inputs/outputs into wire vs
  per-shard-resident cache and pairing `in_*`/`out_*` (the KV loop). Policy, not graph surgery.
- **Transport / discovery / planning** — never tract's job.

Rationale: core stays minimal and model-agnostic; the opinionated parts live where they can
evolve without an API commitment.

## Generality limits (stated honestly)
- **Skip connections spanning >1 stage** → `CrossStageFanout` error; needs boundary-outlet
  promotion to handle (future extension), not silent misbehavior.
- **Arbitrary DAG partition** (not linear pipeline) → use `extract_subgraph` directly with
  hand-chosen inputs/outputs; `split_pipeline` only covers the linear case.
- **A huge const shared across stages** is duplicated per stage (usually the point — that's the
  memory split — but worth documenting).

## Testing / verification (PR-ready)
- **Round-trip parity**: `extract_subgraph`, then run each stage feeding the previous stage's
  captured boundary tensors; assert equal to the whole-model run. Reuse the `--nnef-cycle`
  harness idea (`cli/src/params.rs:794`).
- **Property test** (proptest, `harness/`): a random stateless op chain split at *every* node →
  pipeline output == whole-model output, bit-exact.
- **Guard tests**: cutting a `Scan`/KV output → `StatefulBoundary`; omitting a needed boundary
  input → `UndeclaredBoundaryInput` naming the outlet.
- **Real-model integration**: the openelm 2-stage split already validates prefill + greedy-decode
  parity end-to-end (in `examples/dis-tract/tests/llm_pipeline.rs`); lift as the acceptance case.

## Guard, refined after implementation
Only **one** guard is actually needed in core, and it subsumes the others:
- `extract_subgraph` walks the outputs' dependency cone; any input-less node reached that is
  **not a `Const`** and **not a declared input** is a required boundary the caller omitted → error.
- This single check also catches the **cross-stage skip connection** case: if a residual crossing
  cut `i` is also needed by stage `i+2`, that stage's outputs depend on an outlet not in its
  declared inputs → the same error fires on that stage.
- **Statefulness is NOT a core guard.** Cutting at a stateful op's *tensor* output is safe — the op
  (and its `OpState`/KV) stays wholly in one stage; only the tensor crosses. Which side owns which
  cache is caller *policy* (the role layer in `llm.rs`), deliberately kept out of core.

## Status: implemented + dogfooded
Landed in `core/src/model/partition.rs` (`extract_subgraph`, `split_pipeline`, re-exported from
`tract_core::model`), with 3 unit tests: split-at-every-cut bit-exact on an op chain, prune-to-slice,
and undeclared-input rejection. `tract-distributed` now calls the core functions (its `partition.rs`
→ `split_pipeline`, `llm.rs` → `extract_subgraph`), and the **openelm 2-stage prefill + greedy-decode
parity test passes through the core function** — i.e. the real 16-layer LLM split is the acceptance
test for `extract_subgraph`. This is the whole proposed PR, minus the maintainer decisions below.

## Maintainer questions to settle first
1. Free functions in `core/src/model/` vs methods on `TypedModel` vs a new `partition` module?
2. Is a graph-slicing primitive in-scope for core, or preferred as a `tract-libcli`/example util?
3. Error surface: typed variants (above) vs plain `anyhow` context strings?
```

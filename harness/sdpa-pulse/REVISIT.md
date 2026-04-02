# Things to revisit / generalize later

Items noted during incremental harness development that work correctly for
the current harnesses but may need generalization before the real encoder lands.

---

## 1. `classify_chunk_window` — 2-D window detection

**Location:** `core/src/ops/logic.rs`

**Current state:** Recognises the specific pattern produced by the
`ex04-block-left-1-mask` NNEF graph:
```
Mul([Ge(Val(L), diff), Ge(diff, Val(0))])
where diff = Add([MulInt(-1, Div(🎯1, P)), Div(🎯0, P)])
```

**What may need generalising:**
- The real Nemotron encoder mask uses a different computational path
  (relative-shift trick, different TDim expression trees).  The classifier
  may need to handle additional normal forms of the same logical predicate.
- `uniform_tdim` propagation through the full encoder mask graph
  (range → cast → div → floor → sub → le/ge/and) is the **main unverified
  assumption** of the whole strategy.  If any op in that chain doesn't
  propagate `uniform_tdim`, `FoldWindowAttention` never fires.
- The current classifier is O(1) structural pattern matching; a more robust
  version might canonicalise the expression first (e.g. via TDim
  simplification) before matching.

---

## 2. `PulsedTokenFold` / `PulsedTokenUnfold` — reshape pulsifiers

**Location:** `pulse/src/ops/array/reshape.rs`

**Current state:** Handles the specific case where:
- fold: `AxisOp::Reshape(at, [T_product], [C, P])` with `to.last() == pulse`
- unfold: `AxisOp::Reshape(at, [C, P], [T_product])` with `from.last() == pulse`

Only fires when the reshape axis == streaming axis and the chunk size equals
the pulse size exactly.

**What may need generalising:**
- If the real encoder uses a different reshape order or has extra batch/head
  dims, the axis index assumptions may be wrong.
- The `to_typed()` for `PulsedTokenFold` returns `AxisOp::Add(at)`; this is
  correct for the pulse-time typed model but may interact unexpectedly with
  downstream ChangeAxes optimisations.
- `pulse.to_i64()?` panics (returns Err) if the pulse size is symbolic rather
  than a concrete integer; this is fine for current harnesses but not general.

---

## 3. `change_shape_array` fallback in `ChangeAxes`

**Location:** `core/src/ops/change_axes.rs`

**Current state:** Added a fallback branch that trusts `from.len()` and applies
the reshape when `from_volume == to_volume` (structurally) but the per-element
shape match fails.  This was needed because `S` and `P*(S/P)` are
structurally different TDims even though they're equal when S%P=0.

**What may need generalising / fixing properly:**
- The right fix is an assertion `assert(S % P == 0)` (multiplicity assertion)
  that lets TDim prove `S == P*(S/P)`.  The fallback is a workaround.
- The fallback could in theory fire for genuinely incompatible reshapes if the
  volume check happens to pass structurally — needs a closer look.

---

## 4. `FoldWindowAttention` — rank-2 Q/K/V restriction

**Location:** `core/src/optim/fold_window_attention.rs`

**Current state:** Only handles Q, K, V with rank 2 (`[T, D]`).  The real
encoder has rank 4 (`[B, H, T, D]`).

**What needs to change:**
- Extend to arbitrary rank; the token axis is not necessarily 0.
- The reshape must fold/unfold only the token axis, leaving batch/head axes
  untouched.

---

## 6. `FoldUniformTDim` — dummy-input hack for symbol resolution ordering

**Location:** `core/src/optim/fold_uniform_tdim.rs`

**Current state:** When `FoldUniformTDim` replaces a wire with a `UniformTDim` node
(zero inputs), it wires `model.inputs[0]` as a dummy dependency when the shape
contains model symbols (e.g. S).  This forces `UniformTDim` to be topologically
ordered after the Source node so that S is resolved in `session.resolved_symbols`
before `eval_with_session` tries to evaluate the shape.

**Why it's a hack:**
- It assumes `model.inputs[0]` carries the relevant symbol(s) — true for current
  harnesses but not guaranteed in general (a model may have S derived from input 1,
  or from a shape input that is not `inputs[0]`).
- The right fix is to let `UniformTDim` take the *shape inputs* it actually depends
  on (i.e. the nodes that concretely provide the symbol values), determined by
  tracing which symbols appear in `self.shape` and which source nodes resolve them.
- Alternatively, symbol resolution could be done eagerly at model-load time rather
  than lazily from node outputs, but that would require broader changes to `plan.rs`.

**What to do properly:**
- In `FoldUniformTDim`, collect the symbols appearing in `shape`, find which model
  input outlets (or `ShapeOf` outputs) resolve those symbols, and wire those
  specific outlets as shape-hint inputs to `UniformTDim`.
- Or redesign `UniformTDim` to accept an explicit shape-tensor input (concrete
  at runtime) rather than a symbolic `ShapeFact`.

---

## 5. Pipeline ordering: `FoldWindowAttention` before `ScaledMaskedSoftmax`

**Location:** `core/src/optim/mod.rs` (declutter pass order)

**Current state:** `FoldWindowAttention` runs before `FoldUniformMask` in
`declutter()`.  Not yet verified whether `detect_scaled_masked_softmax`
(in tract-transformers) can fire before FoldWindowAttention on the real
encoder graph and make the pattern unrecognisable.

**Risk:** If `ScaledMaskedSoftmax` fuses the Iff+softmax into a single opaque
op, `FoldWindowAttention` can never fire.  The ordering between
`detect_scaled_masked_softmax` and `FoldWindowAttention` must be verified (or
FoldWindowAttention extended to decompose ScaledMaskedSoftmax inline).

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
  propagate `uniform_tdim`, `FoldUniformMask` never fires.
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

## 8. `Delay` buffer initialisation — zeros vs uninitialized

**Location:** `pulse-opl/src/delay.rs` (`DelayState::eval`)

**Current state:** The Delay buffer is allocated with `Tensor::zero_dt`.
Prefix positions (before the first real input has filled the buffer) are zeroed.

**Why zeroing was chosen:** During ex05 development, the uninitialized buffer
caused NaN outputs on the first pulse of the AV EinSum (`0·NaN = NaN` when the
attention weights multiplied the uninitialized K/V positions).  Switching to
`Tensor::zero_dt` silenced those NaNs.

**Why `uninitialized_dt` would be preferable:** NaN propagation is a feature —
it surfaces incorrect use of prefix outputs (i.e. outputs produced before the delay
has been satisfied) that should have been discarded.  Zeroing hides such bugs silently.

**The actual root cause (ex05):** `pulsify_qk` does not propagate the K Delay's
`stream.delay` to the QK EinSum output (it inherits Q's delay=0 instead).  So the
compare framework accumulates turn 0 output even though K and V are not yet valid.
Two correct fixes:
1. Propagate `max(Q.delay, K.delay)` through QK EinSum → Iff → Softmax → AV EinSum
   to the output, so startup turns are discarded automatically by the framework.
2. Apply the same ChunkWindowMask to V before the AV EinSum, so NaN V values are
   zeroed out before multiplication (IEEE `0*NaN = NaN` is the proximate cause).

`zero_dt` silences the symptom without addressing which outputs should be discarded
or why.  Do not revert to `uninitialized_dt` without a proper fix in place.

---

## 7. `compare --stream` — stitch diagonal sliding-window slices into a matrix

**Location:** `cli/src/compare.rs` (`handle_stream`)

**Current state:** When a pulsed intermediate has a shape structurally incompatible
with the reference (e.g. windowed attention `[P, key_window]` vs full attention
`[S, S]`), the comparison skips that node (marks it unchecked/yellow) rather than
failing.  This is correct but silent.

**What would be better:**
For sliding-window attention intermediates the pulsed slices form a banded diagonal
pattern that can be stitched back into the full `[S, S]` matrix — exactly analogous
to how the simple Delay mechanism stitches `[P, D]` pulses into a `[S, D]` output.
At turn `i` (chunk `c`), the pulsed slice `[P, key_window]` corresponds to
rows `[c*P .. (c+1)*P]` and cols `[(c-L)*P .. (c+1)*P]` of the full matrix.

Implementing this stitching in `handle_stream` would let `compare --stream` verify
the full windowed-attention intermediate matrices, not just the final output,
giving much stronger correctness guarantees for the pulsification of attention.

**Depends on:** knowing the stream axis and the `key_window` / `left_chunks`
metadata for the intermediate — either from pulsed-model facts or from a new
annotation on the accumulated slice.

---

## 5. Pipeline ordering: `ScaledMaskedSoftmax` vs `FoldUniformMask`

**Location:** `core/src/optim/mod.rs` (declutter pass order)

**Risk:** `FoldUniformMask` only handles `Iff` and binary ops with a bool
`uniform_tdim` input.  `ScaledMaskedSoftmax` is opaque to it.  If
`detect_scaled_masked_softmax` (in tract-transformers) fires *before*
pulsification, `FoldUniformMask` cannot fold the mask and the whole strategy breaks.

**Current state for harnesses:** Plain `Iff + softmax` is used, not
`tract_transformers_scaled_masked_softmax`, so `FoldUniformMask` acts directly.

**Generalisation risk:** For the real encoder pipeline, `detect_scaled_masked_softmax`
must run *after* pulsification and mask folding — not before.  This ordering is
currently not enforced.  When wiring up the real encoder pulsification pipeline,
verify the transform order, or extend `FoldUniformMask` to decompose
`ScaledMaskedSoftmax` inline.

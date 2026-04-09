# Things to revisit / generalize later

Items noted during incremental harness development that work correctly for
the current harnesses but may need generalization before the real encoder lands.

---

## 11. `compare --stream` error message — "Undetermined symbol in expression"

**Location:** `cli/src/compare.rs` (or wherever `compare --stream` evaluates the pulsed model)

**Observed:** Running `compare --stream --allow-random-input` on the encoder produces only:

```
ERROR tract] Undetermined symbol in expression: (14)#(15)
```

with no node name, no stack, no indication of which op or wire triggered it,
and `(14)#(15)` is an internal TDim `Broadcast` variant that is opaque to the user.

**What would be better:**
- Report the node name and outlet index where evaluation failed.
- Translate `(N)#(M)` into a human-readable description (e.g. "Broadcast of M and N — symbol not resolved").
- Include the full Caused-by chain (the error is currently swallowed at the compare loop level).

**Why it matters:** Without a node name it is impossible to know whether the failure
is in the pulsed model construction, the reference model evaluation, or the pulse-by-pulse
accumulation loop.  Diagnosing the encoder failure required a bisect + dump workflow
that a better error message would have made unnecessary.

---

## 9. ~~Transformer-XL content-to-position score — Q @ R skew pulsifier~~ ✅ FIXED

**Reproducer:** `harness/sdpa-pulse/ex13-rel-pos-skew-window` (batch + pulse PASS)

**Fix:** Per-operator `input_roi` hooks on Slice, AxisOp::Reshape (with axis-swap),
Pad, DynSlice, and EinSum propagate the chunk-window ROI backward from the attention
mask through the full skew chain.  The Slice pulsifier extends each slice by L*P in
the direction determined by whether `start` decreases with S (center-anchored R
extraction → extend start back) or is fixed (skew slices, pos_scores → extend end
forward).  See `pulse/src/ops/array/slice.rs`.

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

## 10. Unify `Iff` and `ScaledMaskedSoftmax` ROI propagation via `input_roi`

**Location:** `core/src/optim/propagate_roi.rs`, `core/src/ops/logic.rs` (`Iff`)

**Current state:** `PropagateRoi::run_direct` has two separate sub-loops:
1. A hand-coded `Iff`-specific loop with inversion detection (`peel_negated_chunk_window_expr`,
   `peel_condition`, inverted-convention handling).
2. A generic loop that calls `op.input_roi(...)` — currently only `ScaledMaskedSoftmax` overrides this.

**What should happen:** The Iff-specific logic should be migrated into `Iff::input_roi`, making
the hand-coded loop in `PropagateRoi` unnecessary. `PropagateRoi` would then have a single
generic loop over all nodes. This makes every op's ROI contribution operator-local and
removes the asymmetry between `Iff` and `ScaledMaskedSoftmax`.

**Why it wasn't done yet:** `Iff::input_roi` would need to replicate the inversion detection
currently in `PropagateRoi` (walking through `peel_condition`, detecting `extra_inverted`,
deciding which branch is scores vs fill). That logic is subtle and wasn't worth refactoring
during the initial `input_roi` introduction. The two-loop approach is correct but redundant.

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

---

## 13. Systematic `uniform_tdim` propagation

**Location:** `core/src/ops/binary.rs` (output_facts), `pulse/src/ops/binary.rs` (pulsifier)

**Observed:** `uniform_tdim` is set on a few specific ops (Range, comparisons,
UniformTDim) but does not propagate through arithmetic ops like Mul, Sub, Add,
Cast.  For example, `pos_bias = -0.125 * rel_pos` loses `uniform_tdim` at the
Mul because TDim can't represent float scaling.

**Current workaround:** The binary pulsifier in `pulse/src/ops/binary.rs` walks
upstream through scalar-constant TypedBinOp nodes (`find_upstream_uniform_tdim`)
to locate the nearest `uniform_tdim`, then replays the scalar ops in forward
order to recover actual float values (`collect_scalar_op_chain`).

**Proper fix:** Systematic `uniform_tdim` propagation in `output_facts` for all
ops that preserve coordinate structure, analogous to the ROI propagation PR
(#2114).  This would make the upstream-walk workaround unnecessary and ensure
uniform_tdim is available on every wire where the coordinate pattern holds.

---

## 14. `classify_chunk_window` with offset coordinates

**Location:** `core/src/ops/logic.rs` — `extract_div_diff_axes`, `extract_coord_sym_from_div_arg`

**Observed:** After ROI bubbles through Pad/Reshape, coordinate symbols get
offset (`Div(🎯k+1, P)` instead of `Div(🎯k, P)`) and extra `Val` constants
appear in the diff expression.

**Current workaround:** `extract_div_diff_axes` accepts `Div(Add(Sym, Val), P)`
and ignores `Val(_)` terms.  This works because the offsets don't change P, L,
or axis assignment.

**Proper fix:** The ROI bubbling through Pad/Reshape should either normalize
the expression back to canonical form, or the offset should be tracked
explicitly in `ChunkWindowParams` so downstream consumers can use it.
The current approach silently discards the offset information which could
matter for correct coordinate evaluation.

---

## 15. Encoder skew trick: T→P substitution vs pre-sliced r_pos_window

**Location:** `pulse/src/ops/einsum.rs` (pulsify_qk), Slice/DynSlice pulsifiers

**Observed:** For the p1 encoder, pulsify_qk successfully pre-slices
r_pos_proj to [W+P-1, H, Dh] via try_compute_const_with_substitution.
But the downstream skew trick (Pad→Reshape→Slice→Reshape→Slice) uses
T=P from shape_of(q) for its reshape/slice targets, producing [P, 2P-1]
intermediate shapes.  The final Slice (pos_scores = pos_bd[0:T=P])
produces [P, P] instead of [P, W].

The ROI-aware Slice pulsifier would extend [0:P] to [0:W], but the
input (pos_bd) only has 2P-1 columns (from the T=P reshape), so the
bounds check fails and the extension is skipped.

**Root cause:** The skew trick's reshapes use shape_of(q)[streaming_axis]
which becomes P at pulse time.  The pre-sliced r_pos_window has W+P-1
columns, but the reshape to [B, H, -1, T=P] distributes them into
more rows rather than keeping a wider column dimension.

In ex13/ex14 tests, the r_pos is a direct constant (not via EinSum chain),
so pulsify_qk pre-slices before pulsification changes the shape_of chain,
and the downstream skew trick nodes see correct streaming shapes with ROI
extensions.

**Fix options:**
1. Have pulsify_qk wire the entire skew trick chain as a unit, using W
   instead of T for the intermediate shapes
2. Replace shape_of(q) references in the skew trick with values derived
   from the r_pos_window size
3. Add a dedicated SkewTrick composite op that pulsifies as a unit

---

## 16. Pre-flight superlinear-wire check — false-positive warnings

**Location:** `pulse/src/model.rs` — `check_no_unannotated_superlinear_wires`

**Current state:** Before pulsification, every wire whose shape is superlinear
in the streaming symbol (S appears in ≥ 2 dimensions) and has no
`region_of_interest` or `uniform_tdim` gets a `log::warn!`.  This correctly
identifies the ex15/encoder failure (skew trick intermediates, content_scores,
pos_scores all missing ROI).

**Problem:** Working models also trigger warnings on wires that are quadratic
but handled fine by their consumers — e.g. `masked_scores_false_value` (broadcast
fill), `masked_scores` (Iff output), `attn` (softmax output).  These downstream
wires are quadratic but the pulsifiers for Iff/Softmax/AV-EinSum handle them
directly without ROI.

**Proper fix:** Either (a) make the check smarter — e.g. only warn about wires
whose *producers* are not attention-domain ops that handle quadratic output
natively, or (b) fix ROI propagation so that all quadratic wires truly get ROI
(fixing the `bubble_roi` verified-dim mismatch would be a start), then promote
the warning to an error.

**Related:** The `bubble_roi` shape-equality check (`!=` on TDim) rejects
`(S/2)#((S+1)/2)` vs `(S+1)/2` as unequal even though they are semantically
identical given S≥0.  Fixing that would let ROI propagate through Add in
subsampled models and may eliminate most false positives.

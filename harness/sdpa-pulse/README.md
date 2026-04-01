# sdpa-pulse harness

Incremental test cases for pulsifying windowed self-attention, targeting
the Nemotron encoder (`nvidia/nemotron-speech-streaming-en-0.6b`).

Each harness runs in two steps: a batch reference run, then a streaming
compare (`compare --stream`).  The cases build on each other; each one
that passes proves one more piece of the pulsification machinery works.

---

## block-l-eq-p ✓

Chunk-level bidirectional attention with no lookback.  Input `qkv [S, 3P, Dh]`
where S is the chunk count.  Each chunk of P tokens attends only to the other
tokens in the same chunk.  No explicit mask tensor — the block-diagonal structure
comes for free from the EinSum batch axis `c`.

**Proves:**
- Bidirectional within-chunk attention pulsifies trivially.  Each pulse is
  independent; there is no cross-pulse state to carry.
- `tract_assert S>=0` is required so that `min(0, S+1) = 0` simplifies correctly
  in TDim, which the slice deserialization depends on.
- The `ChangeAxes` optimizer legally squeezes the singleton streaming axis from
  intermediate `[1,P,P]` EinSum outputs; `compare --stream` must tolerate that.

**Story role:** Baseline.  Proves the pulsification machinery works at all for
self-attention.  Left-chunk lookback = 0.

---

## block-left-1 ✓

Same chunk-level layout, but each chunk's Q attends to K/V from the previous
chunk as well (left-chunk lookback = 1).  The K/V history is modelled explicitly:
`pad(k, before=1) + slice(end=S)` in the batch graph, which pulsifies to
`Delay(axis=0, delay=1, overlap=0)`.

**Proves:**
- Left-chunk lookback pulsifies via Delay ops.  The pad+slice pattern is exactly
  the unrolled form of a 1-D sliding-window unfold; in streaming the Delay op IS
  the unfold buffer.
- Memory footprint is `(left_chunks+1)*P` K/V vectors per pulse — bounded and
  independent of total sequence length.

**Story role:** First genuinely non-trivial streaming case.  Proves K/V state can
be carried across pulses.  Left-chunk lookback = 1 via explicit windowing.

---

## block-l-eq-p-mask (batch ✓, streaming blocked)

Block-diagonal attention as in `block-l-eq-p`, but now an explicit boolean mask
tensor `[S, P, P]` (all-true) is wired through `select + softmax`.  No lookback.

**Proves (batch only):**
- `Iff + softmax` loads and evaluates correctly with an external boolean mask.

**Where it stops:** The streaming compare requires both `qkv` and `mask` as
per-pulse inputs, but `handle_stream` in the CLI only wires a single input per
pulse.  This is a known limitation of the streaming comparison harness, not of
tract's pulsifier.

**What it does NOT prove:** Because the mask is an external input it has no
`uniform_tdim`.  `FoldUniformMask` never fires.  This test proves that raw `Iff`
pulsifies as an op, not that the mask can be reasoned about structurally.

**Story role:** Stepping stone.  Confirms `Iff + softmax` is wired correctly
before adding a computed mask.  The multi-input streaming issue is a separate
CLI bug to fix.

---

## block-left-1-mask ✓

Flat-token sliding-window attention.  Input `qkv [S, 3Dh]` where S is the
**token** count (not chunk count).  The full T×T attention matrix is computed,
and the mask is computed entirely inside the graph from first principles —
directly adapted from the real Nemotron encoder NNEF:

```
range(0, T) → cast(f32) → div(·, P) → floor → cast(i64)   # chunk index per token
→ unsqueeze [T,1] and [1,T] → sub → diffChunks [T,T]
→ le(·, left_chunks) and ge(·, 0) → chunked_mask [T,T]
```

`T` is derived from `shape_of(qkv)[0]` instead of an external `length` input.
No padding mask.  Pulsify with `--pulse P` (P tokens = 1 chunk per pulse).

**Proves:**
- The Nemotron encoder mask construction can be represented in NNEF and evaluates
  correctly.
- The sliding-window mask is correct: token i attends to token j iff
  `0 <= floor(i/P) - floor(j/P) <= left_chunks`.
- `FoldWindowAttention` detects the `Iff(chunk_window_mask) → Softmax → EinSum`
  pattern from `uniform_tdim` on the mask wire and rewrites it to bounded-window
  chunk attention (`[C, P, D]` layout with explicit K/V context via pad+slice+concat).
- `PulsedTokenFold` / `PulsedTokenUnfold` pulsifiers for `AxisOp::Reshape` correctly
  handle the token-to-chunk fold (`[T, D] → [C, P, D]` at batch time; `[P, D] → [1, P, D]`
  at pulse time) and its inverse.
- Reference semantics: chunk 0's context window is zero-padded (startup latency), which
  is absorbed in streaming by discarding early output pulses via Delay.

**Story role:** The computed-mask milestone.  Proves the full pipeline works end-to-end:
mask construction in NNEF → `FoldWindowAttention` recognises the structure → pulsifier
inserts Delay ops for K/V lookback.

---

## The arc

```
block-l-eq-p        attention pulsifies (trivial, no state)
block-left-1        K/V lookback pulsifies (Delay ops, explicit window)
block-l-eq-p-mask   Iff+softmax pulsifies (external mask, raw op test)
block-left-1-mask   FoldWindowAttention + token fold/unfold pulsifiers  ✓
```

Every passing test proves one more piece of the machinery works.  The failing
test pinpoints what is missing: a `FoldUniformMask` that can recognise a 2-D
chunk-index-based true region and fold the T×T attention down to a bounded
window before the pulsifier runs.

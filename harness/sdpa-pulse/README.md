# sdpa-pulse harness

Incremental test cases for pulsifying windowed self-attention, targeting
the Nemotron encoder (`nvidia/nemotron-speech-streaming-en-0.6b`).

Each harness runs in two steps: a batch reference run, then a streaming
compare (`compare --stream`).  The cases build on each other; each one
that passes proves one more piece of the pulsification machinery works.

---

## ex01-block-l-eq-p ✓

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

## ex03-block-left-1 ✓

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

## ex02-block-l-eq-p-mask (batch ✓, streaming blocked)

Block-diagonal attention as in `ex01-block-l-eq-p`, but now an explicit boolean mask
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

## ex04-block-left-1-mask ✓

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
- `uniform_tdim` propagates through the full mask computation chain
  (`range → cast → div/floor → cast → unsqueeze → sub → le/ge → and`),
  letting `FoldUniformMask` fold the `Iff` nodes away in the pulsed model.
- `ChunkWindowMask` correctly materialises the per-pulse boolean mask
  at streaming time (steady-state: all-true over the `(left_chunks+1)*P` context window).
- Intermediate pulsed shapes (`[P, key_window]`) differ from the reference (`[S, S]`);
  `compare --stream` skips incompatible-shape intermediates rather than failing.
- Reference uses `-inf` masking (natural batch graph semantics); startup latency is
  absorbed in streaming by the Delay discard mechanism.

**Story role:** The computed-mask milestone.  Proves the full `uniform_tdim` propagation
pipeline: mask construction in NNEF → `FoldUniformMask` folds Iff away → pulsifier inserts
Delay ops for K/V lookback.

---

## ex05-block-left-1-posenc ✓

Same sliding-window attention as ex04, plus an ALiBi-style position bias added
to the scores before the mask.  The bias is computed as `−slope × (i − j)` for
each token pair.

**Proves:**
- Position bias (a constant additive term to scores) pulsifies correctly via the
  binary pulsifier (materialised from `region_of_interest` + `uniform_tdim`).
- The full pipeline works with `Add(EinSum_scores, pos_bias)` before the mask,
  without any special handling of the additive term.

**Story role:** Nearest harness approximation to the real Nemotron encoder, which
uses Transformer-XL relative-position attention (content + position scores, both
masked).

---

## ex06-batch-multihead ✓

Same sliding-window attention as ex04, but with batch and head dimensions:
`qkv [1, 2, S, 12]` where axis 2 streams (not axis 0).  Q/K/V are `[1, 2, S, 4]`,
scores and attn are `[1, 2, S, S]`, mask is `[1, 1, S, S]` (broadcast over H).

**Proves:**
- The pulsification machinery handles a non-zero streaming axis (axis 2 of 4).
- `uniform_tdim` propagates correctly through two `unsqueeze` ops, remapping
  coord symbols from `🎯0,🎯1` to `🎯2,🎯3`.
- `classify_chunk_window` recognises the pattern with arbitrary row/col axes.
- The EinSum pulsifiers (`pulsify_qk`, `pulsify_av`) correctly identify Q, K, V
  and their key axes for arbitrary rank via the axes-mapping and streaming fact.
- The Iff pulsifier promotes `ChunkWindowMask`'s rank-2 `[P, kw]` output to
  `[1, 1, P, kw]` by inserting leading `AxisOp::Add(0)` nodes, and creates the
  fill tensor with the matching rank.

**Story role:** Proves the machinery is not rank-2 specific.  Required before
tackling the real encoder's `[B, H, T, T]` attention.

---

## ex07-block-left-1-chunkpos ✓

Same sliding-window attention as ex04, but adds a **chunk-level** relative-position
bias analogous to the Transformer-XL v-bias term:

```
v_bias[i,j] = slope × (floor(i/P) − floor(j/P))   slope = −0.5
```

The bias is zero within the same chunk and `−slope` when j is one chunk earlier.

**Proves:**
- `Div` inside a TDim coordinate expression propagates correctly through
  `uniform_tdim`: the `chunk_diff` wire carries `Div(🎯0, 2) − Div(🎯1, 2)`,
  which the binary pulsifier evaluates at steady-state coords to produce a
  constant `[P, (L+1)*P]` tensor.
- `PropagateRoi` reaches `chunk_diff` through the `Add(scores, pos_bias)` →
  `Mul(chunk_diff, slope)` → `Sub(ci_row, ci_col)` TypedBinOp chain.
- The binary pulsifier correctly handles integer floor-division in the
  coordinate expression — the key step for representing chunk-level position
  encoding without a lookup table.

**What this does NOT cover:** The Transformer-XL Q-dependent content-to-position
score `q[i] @ R[i−j]^T` (which depends on streaming Q).  That term requires
either a dedicated EinSum+gather pulsifier or a rewrite into purely arithmetic
form — a REVISIT item.

**Story role:** Proves `Div` in TDim coordinate expressions works end-to-end.
This is the key building block for any position encoding that is a function of
chunk-index difference (vs token-index difference in ex05).

---

## The arc

```
ex01  block-l-eq-p              attention pulsifies (trivial, no state)                    ✓
ex02  block-l-eq-p-mask         Iff+softmax (external mask, batch only)                    ✓ batch
ex03  block-left-1              K/V lookback (Delay ops, explicit window)                   ✓
ex04  block-left-1-mask         computed mask + uniform_tdim + FoldUniformMask + Delay      ✓
ex05  block-left-1-posenc       ex04 + ALiBi pos bias; binary pulsifier                    ✓
ex06  batch-multihead           ex04 lifted to [B,H,T,T]; streaming axis=2, rank-4         ✓
ex07  block-left-1-chunkpos     chunk-level pos bias; Div() in TDim coord expression       ✓
ex08  batch-mask                [B,S,S] attention with mask                                ✓
ex09  batch-multihead-mask      [B,H,S,S] with mask                                       ✓
ex10  batch-multihead-projections  Q/K/V projections + multihead                           ✓
ex11  batch-scaled-masked-softmax  ScaledMaskedSoftmax op                                  ✓
ex12  rel-pos-skew              skew trick for relative position encoding                  ✓
ex13  rel-pos-skew-window       skew trick + chunk-window mask (DiagGather)                ✓
ex14  rel-pos-skew-large-table  skew trick with oversized position table                   ✓
ex14  reduced-skew              reduced r_pos table via DynSlice                           ✓
ex15  shared-posenc-skew        skew trick + --set S=2*s for verified dim resolution       ✓
ex16  double-subsample-skew     two stride-2 subsamples + skew trick (DiagGather)          ✓
```

Every passing test proves one more piece of the machinery works.

---

## Pulsification semantics

### Goal

Transform a model that consumes a full sequence of length S into one that
processes fixed-size chunks ("pulses") of size P, producing equivalent
output incrementally.

### The increment

Define the **increment** at pulse n as the set of newly computable values:

    delta(n) = Computable(n*P) \ Computable((n-1)*P)

The **pulsed output** of each op is the rectangular hull of delta(n).
Because tensors must be rectangular, the hull may be slightly larger than
delta(n) itself -- the padding is acceptable overhead.

In the classical (linear) case the hull is trivially `[..., P, ...]` --
one pulse-sized axis.

### Per-wire classification

Every wire in a pulsifiable model falls into exactly one category:

1. **Static** -- shape has no dependence on S.  Passes through unchanged.
2. **Streaming-linear** -- exactly one dimension is `a*S + b`.  That
   dimension becomes the pulse axis; the hull per pulse is `a*P + b`.
3. **Streaming-superlinear with ROI** -- multiple dimensions depend on S,
   but a `region_of_interest` annotation proves that the effective
   consumption is linear.  The hull is `[..., P, ..., W, ...]` where W is
   the window width derived from the ROI.  W is constant (independent of n).

Without ROI on category 3, pulsification must refuse.

### The pulsification contract

**A model is pulsifiable iff every wire is either static, linear in S, or
superlinear with an ROI annotation that reduces its hull to bounded size.**

For each op:
- If all I/O are category 1 or 2: classical `pulsify`.
- If some I/O are category 3: the op's pulsifier must understand the ROI
  and produce the windowed hull (e.g. an EinSum with ROI on its output
  computes `[P, W]` instead of `[P, T]`).

### ROI propagation

ROI annotations are established by a backward pass (`PropagateRoi`):

- **Introduction**: ops like `Iff` / `ScaledMaskedSoftmax` read their
  mask's `uniform_tdim` and create an ROI on the scores input.
- **Bubbling**: element-wise ops pass an output ROI through to their inputs
  via `axes_mapping`.
- **Merging**: when multiple consumers produce different ROIs for a wire,
  they are merged via boolean OR (`a + b - a*b`).  If any consumer needs
  all positions (returns `None`), the wire gets no ROI.

The pass iterates to fixpoint.

### Delay buffers

The delay buffer is the portion of `Computable((n-1)*P)` that is still
needed by `delta(n)`.  It is the intersection of the old computable set
with the new dependency set.

For a streaming-linear wire (e.g. a 1-D convolution with kernel K), the
delay is K-1 positions.

For a superlinear wire with ROI (e.g. attention scores `[P, W]`), the key
axis has a delay buffer of `W - P` positions: P new key positions enter the
window each pulse, and P old ones leave.

### The skew trick and DiagGather

Relative position encoding computes `pos_scores[i, k] = q[i] . r[k-i]`,
where `r` is a table of relative-position embeddings.  The "skew trick"
implements this reindexing from relative to absolute coordinates via:

    Pad(pre=1) -> Reshape([T,2T]->[2T,T]) -> Slice(start=1)
               -> Reshape([2T-1,T]->[T,2T-1]) -> Slice(end=T)

Each individual op has complex integer-division indexing, but the
composition is a clean diagonal gather:

    pos_scores[i, k] = pos_raw[i, (T-1) + k - i]

The intermediate reshapes create artificial whole-sequence dependencies
that prevent per-op pulsification.  However, the function's inputs and
outputs both have bounded hulls (the input needs `[P, W+P-1]` relative
positions; the output is `[P, W]`).

`DiagGather` replaces the 5-op chain with a single op whose pulsification
is straightforward: at pulse time, offset becomes `P_local - 1` (where
`P_local` is the streaming pulse at this level) and `out_len` becomes W.
This avoids large pattern matching at pulsification time -- the pattern
is matched once (pre-pulsification fold) and pulsification sees only the
clean semantic op.

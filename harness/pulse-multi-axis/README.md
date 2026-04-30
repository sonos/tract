# pulse-multi-axis harness

Synthetic test cases driving the Blockify graph-rewrite pass for pulse v1.

These are deliberately written *without* attention-specific framing
(no Q/K/V naming, no softmax, no value tensor, no pre-chunked input shape)
so the pulsifier has to discover any per-pulse window structure from the
graph alone.

Each example pairs a `graph.nnef`, a `gen-inputs.py` that emits an
`io.npz` (batch reference), and a `runme.sh` that runs batch and
pulsified via the CLI and asserts both against `io.npz`.  The CLI's
`--assert-output-bundle` path automatically skips `pulse.delay` warmup
tokens, so cases with non-zero output delay (ex03 future-window) align
their streamed output to the batch reference without bespoke Rust glue.

## ex01-block-diag-reduce

Two streams `a, b` of shape `[T, D]`.  Pairwise dot product into a `[T, T]`
score matrix.  Block-diagonal mask (`mask[i,j] = (i/P == j/P)`, P=2)
multiplied in.  Sum-reduce on axis 0 → output `[T]`.

The score matrix wire has streams on **both** of its axes simultaneously.
The block-diagonal mask annihilates everything except the current diagonal
P×P block, which is the structural information that drives Blockify's
rewrite into single-streaming-axis chunked form.

Streaming-axis: 0 on both inputs and on the output.  Pulse size P=2.

`blockified/` contains a hand-written reference of what the post-Blockify
typed graph should look like — model interface preserved (inputs `[2*S, 4]`,
output `[2*S]`), internal reshape factors the streaming axis into chunks.

## ex02-block-diag-bilinear

Same block-diagonal structure as ex01, but the row-axis Reduce<Sum> is
replaced by a second EinSum against a third stream `c [T, D]`:

    output[i, d] = sum_j masked[i, j] * c[j, d]            # [T, D]

This is the SDPA structure (Q·Kᵀ → mask → attn·V) without softmax.
Smallest synthetic that exercises a downstream second EinSum after the
masked score matrix.

Blockify recognises this pattern (the recogniser matches Mul-by-mask
followed by either `Reduce<Sum>` or a contracting `EinSum`) and rewrites
the second EinSum the same way as the first, with the chunk batch axis
prepended to its subscripts.  Numerical match is verified end-to-end in
`harness/core-proptest-pulse/tests/blockify_ex01.rs`.

## ex03-banded-reduce

Same shape as ex01, but with an **asymmetric** mask — every row at chunk
`c` attends to chunks `{c, c-1, ..., c-L}` (here `L = 1`).  The mask
matrix is a P-block lower bidiagonal, mimicking the geometry of
multi-chunk attention with left-context.

The structural justification for chunked pulsification is identical to
ex01 (per-pulse work bounded by `(L+1)·P` past samples plus the current
P-block).  Only the mask predicate differs: `eq(chunk_row, chunk_col)`
in ex01 becomes `0 ≤ chunk_row - chunk_col ≤ L` here.

Blockify recognises the banded form `(diff >= L1) && (diff <= L2)` —
parametrised by `MaskForm::Banded { lower, upper, k }` — and rewrites
the contracted-axis input by wrapping it with `WindowOnAxis(W)` (where
`W = upper − lower + 1`) followed by a flatten reshape, so the chunked
einsum's contracted axis carries `W·k` elements per chunk instead of
`k`.  Pulsification is non-causal: the streamed output is delayed by
`L = upper − lower` chunks (the future-lookahead the band requires),
flushed by feeding zero-chunks at the end of the stream.  Numerical
match is verified end-to-end in
`harness/core-proptest-pulse/tests/blockify_ex01.rs::ex03_*`.

## ex04-banded-causal

Same shape as ex03 but with a *causal* mask: every row at chunk `c`
attends only to chunks `{c, c-1, ..., c-L}` (no future lookahead).
Mask form: `-L ≤ diff ≤ 0` with `diff = chunk(i) - chunk(j)`.  For
`L = 1, P = 2` the mask is the upper P-block bidiagonal (mirror of
ex03).

Blockify dispatches the same banded recogniser, but `WindowOnAxis` is
parameterised with `start = lower < 0` (past-window flavour).  The
pulsifier wires `Delay(0, W-1) → PulsePad(before = -start) →
PulsedExposeWindow`: the `PulsePad` zero-fills the leading `-start`
chunks of the post-Delay buffer (matching the out-of-stream zero
semantics of the batch reference) and shifts `stream.delay` back so
the final output has `stream.delay = 0` — fully causal, no trailing
flush.  Numerical match in `ex04_*`.

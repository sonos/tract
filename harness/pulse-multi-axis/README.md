# pulse-multi-axis harness

Synthetic test cases driving the Blockify graph-rewrite pass for pulse v1.

These are deliberately written *without* attention-specific framing
(no Q/K/V naming, no softmax, no value tensor, no pre-chunked input shape)
so the pulsifier has to discover any per-pulse window structure from the
graph alone.

Each example is two-step: a batch reference run, then a streaming compare
(`compare --stream`).

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

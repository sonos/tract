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

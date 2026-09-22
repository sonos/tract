#!/usr/bin/env python3
"""Generate io.npz for the batchify move-axis synthetic.

The model's own layout is `[T, B, 4]`; `io.npz` is in that layout and
`io-batch-first.npz` is the same numbers with the batch axis brought to the
front, which is what the model batchify returns reads and answers.
"""

import numpy as np

T, B = 5, 2
rng = np.random.default_rng(42)

w = np.array(
    [[[1.0, 0.5, -0.25], [0.0, 1.0, 0.5], [-0.5, 0.25, 1.0], [0.75, -1.0, 0.25]]],
    dtype=np.float32,
)

x = rng.standard_normal((T, B, 4)).astype(np.float32)
output = (x @ w).sum(axis=2, keepdims=True).astype(np.float32)

np.savez("io.npz", x=x, output=output)
np.savez(
    "io-batch-first.npz",
    x=x.transpose(1, 0, 2).copy(),
    output=output.transpose(1, 0, 2).copy(),
)
print("Saved io.npz and io-batch-first.npz")

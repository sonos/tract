#!/usr/bin/env python3
"""Generate io.npz for the laned causal-conv synthetic.

One row of batch -- the batch axis is the lane axis, so a stream feeds one row
per turn -- and a stream long enough to hold several turns: a laned run seats
the streams at different positions in it, which is what makes a piece of state
two streams share show up as a diff.

Parameters
----------
B = 1 (batch, i.e. one row per stream)
P = 2 (pulse)
T = 16 (stream length)
"""

import numpy as np

B, P, T = 1, 2, 16

rng = np.random.default_rng(42)

a = rng.standard_normal((B, 1, T)).astype(np.float32)

kernel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
padded = np.pad(a, ((0, 0), (0, 0), (2, 0)))
output = np.stack(
    [np.correlate(row, kernel, mode="valid") for row in padded[:, 0, :]]
).reshape(B, 1, T).astype(np.float32)

np.savez("io.npz", a=a, output=output)
print(f"Saved io.npz  a={a.shape} output={output.shape}")

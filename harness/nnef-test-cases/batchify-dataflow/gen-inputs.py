#!/usr/bin/env python3
"""Generate io.npz and io-b2.npz for the batchify synthetic.

`io.npz` is the model as exported, one stream of T frames. `io-b2.npz` is the
same model batchified to two seats, and the two seats carry *different* frames
on purpose: a batchify that smears one seat's values into another passes an
equal-seats bundle and fails this one. Each seat's expectation is the
single-stream computation on that seat's frames alone.
"""

import numpy as np

T = 5
rng = np.random.default_rng(42)

w = np.array(
    [[1.0, 0.5, -0.25], [0.0, 1.0, 0.5], [-0.5, 0.25, 1.0], [0.75, -1.0, 0.25]],
    dtype=np.float32,
)
bias = np.array([[0.5, -0.25, 1.0]], dtype=np.float32)
scale = np.array([[0.5, 2.0, -1.5]], dtype=np.float32)


def network(x):
    act = np.tanh((x @ w + bias) * scale)
    return act.sum(axis=1, keepdims=True).astype(np.float32)


x = rng.standard_normal((T, 4)).astype(np.float32)
np.savez("io.npz", x=x, scale=scale, output=network(x))

seats = rng.standard_normal((2, T, 4)).astype(np.float32)
np.savez(
    "io-b2.npz",
    x=seats,
    scale=scale,
    output=np.stack([network(seat) for seat in seats]),
)
print("Saved io.npz and io-b2.npz")

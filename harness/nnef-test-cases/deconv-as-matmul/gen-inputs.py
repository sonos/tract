#!/usr/bin/env python3
"""Weights and io.npz for the deconv-as-matmul case.

A 64 -> 64 channel transposed convolution, kernel 8 at stride 4, padding
(2, 2), with a bias, over 6 input frames.  The reference is numpy's
overlap-add of each frame's kernel product.
"""

import nnef
import numpy as np

CI, CO, K, S, T, PAD = 64, 64, 8, 4, 6, 2

rng = np.random.default_rng(7)
kernel = (rng.standard_normal((CO, CI, K)) / np.sqrt(CI)).astype(np.float32)
bias = rng.standard_normal((1, CO)).astype(np.float32)
x = rng.standard_normal((1, CI, T)).astype(np.float32)

full = np.zeros((1, CO, (T - 1) * S + K), np.float32)
for t in range(T):
    full[0, :, t * S : t * S + K] += np.einsum("oik,i->ok", kernel, x[0, :, t])
output = full[:, :, PAD : full.shape[2] - PAD] + bias[:, :, None]

with open("kernel.dat", "wb") as f:
    nnef.write_tensor(f, kernel)
with open("bias.dat", "wb") as f:
    nnef.write_tensor(f, bias)
np.savez("io.npz", input=x, output=output)
print("input", x.shape, "output", output.shape)

//! Shared test-data generator for the gated-delta-net / causal-conv1d-update
//! op tests.

#![cfg(test)]

use tract_nnef::internal::*;

/// Simple deterministic pseudo-random floats in `[-1, 1]`.
pub(super) fn arb(shape: &[usize], seed: u64) -> Tensor {
    let len: usize = shape.iter().product();
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let data: Vec<f32> = (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect();
    Tensor::from_shape(shape, &data).unwrap()
}

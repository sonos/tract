//! Synthetic models for the M1 smoke test (no external assets).

use anyhow::Result;
use tract_core::ops::einsum::EinSum;
use tract_core::prelude::*;

/// Deterministic varied fill in [-1, 1) so results are reproducible across runs.
fn fill(shape: &[usize], seed: usize) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> =
        (0..n).map(|i| (((i * 13 + seed * 7) % 29) as f32 - 14.0) / 14.0).collect();
    Tensor::from_shape(shape, &data).unwrap()
}

fn matmul() -> EinSum {
    EinSum { axes: "mk,kn->mn".parse().unwrap(), operating_dt: f32::datum_type(), q_params: None }
}

/// `x[1,in] · W1 → sigmoid("act") → · W2 → y[1,out]`. The `"act"` node is the
/// intended pipeline cut point.
pub fn build_mlp(in_dim: usize, hidden: usize, out_dim: usize) -> Result<TypedModel> {
    let mut m = TypedModel::default();
    let x = m.add_source("x", f32::fact([1, in_dim]))?;
    let w1 = m.add_const("w1", fill(&[in_dim, hidden], 1).into_arc_tensor())?;
    let h = m.wire_node("mm1", matmul(), &[x, w1])?[0];
    let a = m.wire_node("act", tract_core::ops::nn::sigmoid(), &[h])?[0];
    let w2 = m.add_const("w2", fill(&[hidden, out_dim], 2).into_arc_tensor())?;
    let y = m.wire_node("mm2", matmul(), &[a, w2])?[0];
    m.select_output_outlets(&[y])?;
    m.into_decluttered()
}

/// A reproducible input for [`build_mlp`].
pub fn mlp_input(in_dim: usize) -> Tensor {
    fill(&[1, in_dim], 42)
}

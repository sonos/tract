//! Shared f32 round-trip helpers for CPU-fallback recurrence ops
//! (gated-delta-net, causal-conv1d-update): both compute in f32 internally
//! regardless of input dtype, casting back to the original dtype on output.

use tract_nnef::internal::*;

pub(super) fn to_f32_vec(t: &TValue) -> TractResult<Vec<f32>> {
    let cow = t.cast_to::<f32>()?;
    Ok(cow.to_plain_array_view::<f32>()?.iter().copied().collect())
}

pub(super) fn from_f32(data: Vec<f32>, shape: &[usize], dt: DatumType) -> TractResult<Tensor> {
    let t = Tensor::from_shape(shape, &data)?;
    Ok(t.cast_to_dt(dt)?.into_owned())
}

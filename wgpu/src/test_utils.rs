#![cfg(test)]

use tract_core::internal::*;
use tract_core::transform::ModelTransform;

use crate::WgpuTransform;

pub(crate) fn close(a: &Tensor, b: &Tensor) -> TractResult<()> {
    a.close_enough(b, Approximation::Close)
}

/// The suites check what a fused graph computes; these check that it fused.
pub(crate) fn transformed(mut model: TypedModel) -> TractResult<TypedModel> {
    crate::context::wgpu_context();
    WgpuTransform.transform(&mut model)?;
    model.into_optimized()
}

pub(crate) fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 6.0 - 3.0
        })
        .collect()
}

pub(crate) fn gpu_vs_cpu(model: TypedModel, input: Tensor) -> TractResult<()> {
    crate::context::wgpu_context();
    let cpu = SimplePlan::new(model.clone())?;
    let rt =
        tract_core::runtime::runtime_for_name("wgpu")?.context("wgpu runtime not registered")?;
    let gpu = rt.prepare(model)?;
    let g = gpu.run(tvec!(input.clone().into()))?.remove(0).into_tensor();
    let c = Arc::new(cpu).run(tvec!(input.into()))?.remove(0).into_tensor();
    g.close_enough(&c, Approximation::Approximate)
}

#[derive(Debug)]
pub(crate) struct OneWeight(pub(crate) Tensor);

impl WeightSource for OneWeight {
    fn load(&self, _name: &str, _dt: DatumType, _shape: &[usize]) -> TractResult<Tensor> {
        Ok(self.0.clone())
    }
}

/// Runs the model `build` makes in f16 on the GPU and in f32 on the CPU.
pub(crate) fn f16_vs_f32(
    build: impl Fn(DatumType) -> TractResult<TypedModel>,
    input: Tensor,
    approx: Approximation,
) -> TractResult<()> {
    let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
    let gpu = rt.prepare(build(f16::datum_type())?)?;
    let g = gpu.run(tvec!(input.cast_to::<f16>()?.into_owned().into()))?.remove(0);
    let cpu = Arc::new(SimplePlan::new(build(f32::datum_type())?)?);
    let c = cpu.run(tvec!(input.cast_to::<f16>()?.cast_to::<f32>()?.into_owned().into()))?;
    let c = c[0].cast_to::<f16>()?.into_owned();
    g.into_tensor().close_enough(&c, approx)
}

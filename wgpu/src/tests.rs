#![cfg(test)]

use tract_core::internal::*;
use tract_core::ops::math::mul;
use tract_core::transform::ModelTransform;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::kernels::cast::wgpu_cast_dispatch;
use crate::kernels::copy::wgpu_copy_nd_dispatch;
use crate::{WgpuTransform, with_wgpu_queue};

fn close(a: &Tensor, b: &Tensor) -> TractResult<()> {
    a.close_enough(b, Approximation::Close)
}

#[test]
fn upload_download_roundtrip() -> TractResult<()> {
    with_wgpu_queue(|_| {
        let t = Tensor::from_shape(&[2, 3], &(0..6).map(|i| i as f32).collect::<Vec<_>>())?;
        let d = t.clone().into_device()?;
        let back = d.to_host()?.into_tensor();
        assert_eq!(t, back);
        Ok(())
    })
}

#[test]
fn copy_nd_transposed_matches_cpu() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let t = Tensor::from_shape(&[2, 3], &(0..6).map(|i| i as f32).collect::<Vec<_>>())?;
        let src = t.clone().into_device()?;
        let dst = DeviceTensor::uninitialized_dt(DatumType::F32, &[3, 2])?;
        wgpu_copy_nd_dispatch(&src, 0, &[1, 3], &dst, 0, &[3, 2], &[2, 1])?;
        q.flush()?;
        close(&dst.to_host()?.into_tensor(), &t.clone().permute_axes(&[1, 0])?)
    })
}

#[test]
fn cast_f32_f16_roundtrip() -> TractResult<()> {
    with_wgpu_queue(|q| {
        if !q.context().shader_f16() {
            return Ok(());
        }
        let t = Tensor::from_shape(&[4], &[1.0f32, -2.5, 0.25, 100.0])?;
        let src = t.clone().into_device()?;
        let half = DeviceTensor::uninitialized_dt(DatumType::F16, &[4])?;
        let back = DeviceTensor::uninitialized_dt(DatumType::F32, &[4])?;
        wgpu_cast_dispatch(&src, &half)?;
        wgpu_cast_dispatch(&half, &back)?;
        q.flush()?;
        close(&back.to_host()?.into_tensor(), &t)
    })
}

#[test]
fn coverage_accepts_fully_translated_model() -> TractResult<()> {
    crate::context::wgpu_context();
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4]))?;
    let two = model.add_const("two", tensor2(&[[2.0f32]]))?;
    let y = model.wire_node("mul", mul(), &[x, two])?[0];
    model.select_output_outlets(&[y])?;
    WgpuTransform.transform(&mut model)?;
    model = model.into_optimized()?;
    crate::ensure_wgpu_coverage(&model)?;
    Ok(())
}

#[test]
fn hybrid_logsoftmax_matches_cpu() -> TractResult<()> {
    crate::context::wgpu_context();
    ensure!(
        crate::hybrid_fallback_available(),
        "native hybrid fallback should always be available"
    );
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4]))?;
    let y = model.wire_node(
        "attn_sm",
        tract_core::ops::nn::Softmax::new(
            tvec![1],
            None,
            tract_core::ops::nn::SoftmaxKind::LogSoftmax,
        ),
        &[x],
    )?[0];
    model.select_output_outlets(&[y])?;
    let cpu_model = model.clone();
    let rt =
        tract_core::runtime::runtime_for_name("wgpu")?.context("wgpu runtime not registered")?;
    let gpu = rt.prepare(model)?;
    let cpu_plan = SimplePlan::new(cpu_model)?;
    let input = Tensor::from_shape(&[2, 4], &[0.1f32, 0.2, 0.3, 0.4, -1.0, 0.0, 1.0, 2.0])?;
    let gpu_out = gpu.run(tvec!(input.clone().into()))?.remove(0).into_tensor();
    let cpu_out = Arc::new(cpu_plan).run(tvec!(input.into()))?.remove(0).into_tensor();
    close(&gpu_out, &cpu_out)
}

#[test]
fn coverage_rejects_logsoftmax_by_name() -> TractResult<()> {
    crate::context::wgpu_context();
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4]))?;
    let y = model.wire_node(
        "attn_sm",
        tract_core::ops::nn::Softmax::new(
            tvec![1],
            None,
            tract_core::ops::nn::SoftmaxKind::LogSoftmax,
        ),
        &[x],
    )?[0];
    model.select_output_outlets(&[y])?;
    WgpuTransform.transform(&mut model)?;
    model = model.into_optimized()?;
    let err = crate::ensure_wgpu_coverage(&model).expect_err("logsoftmax must be uncovered");
    let msg = format!("{err:?}");
    ensure!(msg.contains("LogSoftmax"), "error should name LogSoftmax, got {msg}");
    ensure!(msg.contains("attn_sm"), "error should name the node, got {msg}");
    Ok(())
}

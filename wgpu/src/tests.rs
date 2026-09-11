#![cfg(test)]

use tract_core::internal::*;
use tract_core::ops::math::mul;
use tract_core::transform::ModelTransform;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::kernels::cast::wgpu_cast_dispatch;
use crate::kernels::copy::wgpu_copy_nd_dispatch;
use crate::kernels::element_wise::wgpu_element_wise_dispatch;
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

/// The suites check what a fused graph computes; these check that it fused.
fn transformed(mut model: TypedModel) -> TractResult<TypedModel> {
    crate::context::wgpu_context();
    WgpuTransform.transform(&mut model)?;
    model.into_optimized()
}

#[test]
fn elementwise_run_becomes_one_chain() -> TractResult<()> {
    use tract_core::ops::nn::sigmoid;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4]))?;
    let a = model.wire_node("a", sigmoid(), &[x])?[0];
    let b = model.wire_node("b", tract_core::ops::math::tanh(), &[a])?[0];
    let c = model.wire_node("c", sigmoid(), &[b])?[0];
    model.select_output_outlets(&[c])?;
    let model = transformed(model)?;
    let chains = model
        .nodes()
        .iter()
        .filter_map(|n| n.op_as::<crate::ops::chain::WgpuElementWiseChain>())
        .collect::<Vec<_>>();
    ensure!(chains.len() == 1, "three element-wise ops should leave one chain, got {chains:?}");
    ensure!(chains[0].steps.len() == 3, "chain should hold all three steps");
    Ok(())
}

#[test]
fn bias_and_activation_become_a_gemm_epilogue() -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::nn::sigmoid;
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact([6, 5]))?;
    let b = model.add_const("b", Tensor::zero::<f32>(&[5, 7])?)?;
    let bias = model.add_const("bias", Tensor::zero::<f32>(&[1, 7])?)?;
    let mm = model.wire_node(
        "mm",
        PrefixMatMul {
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            quantize_output: None,
            operating_dt: None,
        },
        &[a, b],
    )?[0];
    let biased = model.wire_node("bias_add", tract_core::ops::math::add(), &[mm, bias])?[0];
    let y = model.wire_node("act", sigmoid(), &[biased])?[0];
    model.select_output_outlets(&[y])?;
    let model = transformed(model)?;
    let gemms = model
        .nodes()
        .iter()
        .filter_map(|n| n.op_as::<crate::ops::matmul::WgpuGemm>())
        .collect::<Vec<_>>();
    ensure!(gemms.len() == 1, "expected one gemm, got {}", gemms.len());
    ensure!(
        gemms[0].epilogue.len() == 2,
        "bias and activation should ride the gemm, got {:?}",
        gemms[0].epilogue
    );
    Ok(())
}

#[test]
fn a_move_rides_the_op_that_reads_it() -> TractResult<()> {
    use tract_core::ops::nn::sigmoid;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 3, 4]))?;
    let m = model.wire_node("mv", AxisOp::Move(0, 2), &[x])?[0];
    let y = model.wire_node("act", sigmoid(), &[m])?[0];
    model.select_output_outlets(&[y])?;
    let model = transformed(model)?;
    let fused = model
        .nodes()
        .iter()
        .filter(|n| n.op_is::<crate::ops::fused_axis_op::WgpuFusedAxisOp>())
        .count();
    ensure!(fused == 1, "the move should ride its consumer, got {fused} fused nodes");
    Ok(())
}

/// WebGPU caps a dispatch dimension at 65535 workgroups, which at 64 threads
/// each is 4_194_240 values — a feature map a real model reaches. The excess
/// rides the y axis, so the kernel has to fold it back in.
#[test]
fn copy_past_one_grid_dimension() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let n = crate::kernels::shaders::GRID_LIMIT as usize
            * crate::kernels::shaders::WORKGROUP as usize
            + 1000;
        let t =
            Tensor::from_shape(&[2, n / 2], &(0..n).map(|i| (i % 97) as f32).collect::<Vec<_>>())?;
        let src = t.clone().into_device()?;
        let dst = DeviceTensor::uninitialized_dt(DatumType::F32, &[n / 2, 2])?;
        wgpu_copy_nd_dispatch(&src, 0, &[1, (n / 2) as isize], &dst, 0, &[n / 2, 2], &[2, 1])?;
        q.flush()?;
        close(&dst.to_host()?.into_tensor(), &t.permute_axes(&[1, 0])?)
    })
}

/// The element-wise kernels are the ones a large feature map actually reaches,
/// and they take four values a thread, so the grid runs out four times later.
#[test]
fn element_wise_past_one_grid_dimension() -> TractResult<()> {
    use tract_core::ops::nn::sigmoid;
    with_wgpu_queue(|q| {
        let n = 4
            * crate::kernels::shaders::GRID_LIMIT as usize
            * crate::kernels::shaders::WORKGROUP as usize
            + 4000;
        let data = (0..n).map(|i| ((i % 17) as f32) - 8.0).collect::<Vec<_>>();
        let t = Tensor::from_shape(&[n], &data)?;
        let cpu = sigmoid()
            .eval_out_of_plan(tvec!(t.clone().into_tvalue()))?
            .unwrap()
            .remove(0)
            .into_tensor();
        let input = t.into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F32, input.shape())?;
        wgpu_element_wise_dispatch(&*sigmoid().0, &input, &output)?;
        q.flush()?;
        close(&output.to_host()?.into_tensor(), &cpu)
    })
}
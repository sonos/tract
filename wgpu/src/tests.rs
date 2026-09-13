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

#[test]
fn rgba8_ingest_to_nchw_f32() -> TractResult<()> {
    with_wgpu_queue(|q| {
        // 2x2 packed RGBA8. textureLoad of rgba8unorm yields 0..1 floats.
        let rgba: [u8; 16] = [
            255, 0, 0, 255, // (0,0) red
            0, 255, 0, 255, // (1,0) green
            0, 0, 255, 255, // (0,1) blue
            255, 255, 255, 255, // (1,1) white
        ];
        let gpu = crate::tensor_from_rgba8(2, 2, &rgba)?;
        q.flush()?;
        let host = gpu.to_host()?.into_tensor();
        let expected = Tensor::from_shape(
            &[1, 3, 2, 2],
            &[
                1.0f32, 0.0, 0.0, 1.0, // R
                0.0, 1.0, 0.0, 1.0, // G
                0.0, 0.0, 1.0, 1.0, // B
            ],
        )?;
        close(&host, &expected)
    })
}

// --- fusion arithmetic: the fused graph must compute what the ops did apart ---

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32) * 6.0 - 3.0
        })
        .collect()
}

fn gpu_vs_cpu(model: TypedModel, input: Tensor) -> TractResult<()> {
    crate::context::wgpu_context();
    let cpu = SimplePlan::new(model.clone())?;
    let rt =
        tract_core::runtime::runtime_for_name("wgpu")?.context("wgpu runtime not registered")?;
    let gpu = rt.prepare(model)?;
    let g = gpu.run(tvec!(input.clone().into()))?.remove(0).into_tensor();
    let c = Arc::new(cpu).run(tvec!(input.into()))?.remove(0).into_tensor();
    g.close_enough(&c, Approximation::Approximate)
}

/// Wires a random element-wise op, matching the operand's own rank for any
/// constant it needs -- a rank-0 scalar against a rank-N tensor is rejected
/// by tract's typed fact check before this can even reach the GPU.
fn wire_ew(model: &mut TypedModel, name: &str, code: u8, w: OutletId) -> TractResult<OutletId> {
    use tract_core::ops::math;
    let rank = model.outlet_fact(w)?.rank();
    let scalar = |model: &mut TypedModel, nm: String, v: f32| -> TractResult<OutletId> {
        model.add_const(nm, Tensor::from_shape(&vec![1usize; rank], &[v])?)
    };
    Ok(match code % 8 {
        0 => model.wire_node(name, tract_core::ops::nn::sigmoid(), &[w])?[0],
        1 => model.wire_node(name, math::tanh(), &[w])?[0],
        2 => model.wire_node(name, math::abs(), &[w])?[0],
        3 => model.wire_node(name, math::neg(), &[w])?[0],
        4 => model.wire_node(name, math::square(), &[w])?[0],
        5 => {
            let k = scalar(model, format!("{name}.k"), 0.5)?;
            model.wire_node(name, math::add(), &[w, k])?[0]
        }
        6 => {
            let k = scalar(model, format!("{name}.k"), 1.25)?;
            model.wire_node(name, math::mul(), &[w, k])?[0]
        }
        _ => {
            let z = scalar(model, format!("{name}.k"), 0.0)?;
            model.wire_node(name, math::max(), &[w, z])?[0]
        }
    })
}

fn check_ew_chain(len: usize, ops: &[u8], seed: u64) -> TractResult<()> {
    let mut model = TypedModel::default();
    let mut w = model.add_source("x", f32::fact([len]))?;
    for (i, op) in ops.iter().enumerate() {
        w = wire_ew(&mut model, &format!("e{i}"), *op, w)?;
    }
    model.select_output_outlets(&[w])?;
    gpu_vs_cpu(model, Tensor::from_shape(&[len], &fill(seed, len))?)
}

fn check_move_then_chain(
    shape: [usize; 3],
    from: usize,
    to: usize,
    ops: &[u8],
    seed: u64,
) -> TractResult<()> {
    let mut model = TypedModel::default();
    let mut w = model.add_source("x", f32::fact(&shape))?;
    if from != to {
        w = model.wire_node("mv", AxisOp::Move(from, to), &[w])?[0];
    }
    for (i, op) in ops.iter().enumerate() {
        w = wire_ew(&mut model, &format!("e{i}"), *op, w)?;
    }
    model.select_output_outlets(&[w])?;
    let n: usize = shape.iter().product();
    gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(seed, n))?)
}

fn check_gemm_epilogue(
    m: usize,
    k: usize,
    n: usize,
    per_col: bool,
    act: u8,
    seed: u64,
) -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact([m, k]))?;
    let b = model.add_const("b", Tensor::from_shape(&[k, n], &fill(seed ^ 1, k * n))?)?;
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
    let bias_shape: Vec<usize> = if per_col { vec![1, n] } else { vec![1, 1] };
    let bias_len: usize = bias_shape.iter().product();
    let bias =
        model.add_const("bias", Tensor::from_shape(&bias_shape, &fill(seed ^ 2, bias_len))?)?;
    let mut y = model.wire_node("bias_add", tract_core::ops::math::add(), &[mm, bias])?[0];
    if act != 0 {
        y = wire_ew(&mut model, "act", act, y)?;
    }
    model.select_output_outlets(&[y])?;
    gpu_vs_cpu(model, Tensor::from_shape(&[m, k], &fill(seed, m * k))?)
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig::with_cases(24))]

    /// Bias and a trailing activation ride the GEMM as its epilogue; the
    /// epilogue arithmetic must match add-then-activate on the result. This
    /// is what caught the tanh overflow this commit fixes: any activation
    /// drawn against a matmul-scale accumulator exercises the full range a
    /// standalone small-tensor test never reaches.
    #[test]
    fn prop_gemm_epilogue_matches_cpu(
        m in 1usize..9, k in 1usize..9, n in 1usize..9,
        per_col in proptest::prelude::any::<bool>(),
        act in 0u8..8,
        seed in proptest::prelude::any::<u64>(),
    ) {
        check_gemm_epilogue(m * 4, k * 4, n * 4, per_col, act, seed).unwrap();
    }

    /// A run of element-wise ops fuses to one chain kernel; it must still
    /// compute the composition.
    #[test]
    fn prop_elementwise_chain_matches_cpu(
        len in 1usize..96,
        ops in proptest::collection::vec(0u8..8, 2..7),
        seed in proptest::prelude::any::<u64>(),
    ) {
        check_ew_chain(len, &ops, seed).unwrap();
    }

    /// A shape-only Move in front of an element-wise chain becomes a strided
    /// read; the values must land where the copy would have put them.
    #[test]
    fn prop_axis_move_then_chain_matches_cpu(
        d0 in 1usize..6, d1 in 1usize..6, d2 in 1usize..6,
        from in 0usize..3, to in 0usize..3,
        ops in proptest::collection::vec(0u8..8, 1..4),
        seed in proptest::prelude::any::<u64>(),
    ) {
        check_move_then_chain([d0, d1, d2], from, to, &ops, seed).unwrap();
    }
}

/// A sum over both trailing axes at once, as a global pooling is: one
/// cooperative launch instead of one per axis.
#[test]
fn trailing_sum_run_matches_cpu() -> TractResult<()> {
    use tract_core::ops::nn::{Reduce, Reducer};
    for shape in [[3, 20, 17], [16, 9, 16], [2, 1000, 3]] {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact(shape))?;
        let y =
            model.wire_node("sum", Reduce { axes: tvec![1, 2], reducer: Reducer::Sum }, &[a])?[0];
        model.select_output_outlets(&[y])?;
        let n: usize = shape.iter().product();
        gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(7, n))?)?;
    }
    Ok(())
}

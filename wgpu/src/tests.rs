#![cfg(test)]

use tract_core::internal::*;
use tract_core::ops::math::mul;
use tract_core::transform::ModelTransform;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};

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

/// A transpose with ragged tiles on both of its transposed axes.
#[test]
fn copy_nd_tiled_transpose_matches_cpu() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let (b, r, c) = (3, 45, 70);
        let t =
            Tensor::from_shape(&[b, r, c], &(0..b * r * c).map(|i| i as f32).collect::<Vec<_>>())?;
        let src = t.clone().into_device()?;
        let dst = DeviceTensor::uninitialized_dt(DatumType::F32, &[b, c, r])?;
        let (r, c) = (r as isize, c as isize);
        wgpu_copy_nd_dispatch(
            &src,
            0,
            &[r * c, 1, c],
            &dst,
            0,
            &[b, c as usize, r as usize],
            &[r * c, r, 1],
        )?;
        q.flush()?;
        close(&dst.to_host()?.into_tensor(), &t.permute_axes(&[0, 2, 1])?)
    })
}

/// More transpose tiles than one grid dimension holds, folded into a second.
#[test]
fn copy_nd_tiled_transpose_past_one_grid_dimension() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let (rows, cols) = (32, 2_097_184);
        let t = Tensor::from_shape(
            &[cols, rows],
            &(0..rows * cols).map(|i| (i % 251) as f32).collect::<Vec<_>>(),
        )?;
        let src = t.clone().into_device()?;
        let dst = DeviceTensor::uninitialized_dt(DatumType::F32, &[rows, cols])?;
        wgpu_copy_nd_dispatch(
            &src,
            0,
            &[1, rows as isize],
            &dst,
            0,
            &[rows, cols],
            &[cols as isize, 1],
        )?;
        q.flush()?;
        close(&dst.to_host()?.into_tensor(), &t.permute_axes(&[1, 0])?)
    })
}

/// Nine axes, more than the shader takes, of which the extent-one ones merge
/// away.
#[test]
fn copy_nd_merges_past_the_shader_rank() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let shape = [2, 1, 3, 1, 1, 2, 1, 4, 5];
        let t = Tensor::from_shape(&shape, &(0..240).map(|i| i as f32).collect::<Vec<_>>())?;
        let src = t.clone().into_device()?;
        let dst = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
        wgpu_copy_nd_dispatch(&src, 0, src.strides(), &dst, 0, &shape, dst.strides())?;
        q.flush()?;
        close(&dst.to_host()?.into_tensor(), &t)
    })
}

/// A slice of a rank-7 tensor merges down to what the copy addresses, so it
/// stays on device.
#[test]
fn slice_rank_7_device_tensor() -> TractResult<()> {
    with_wgpu_queue(|_| {
        let input = Tensor::from_shape(
            &[2, 1, 3, 1, 4, 5, 6],
            &(0..720).map(|i| i as f32).collect::<Vec<_>>(),
        )?;
        let sliced = input.clone().into_device()?.into_tensor().slice(4, 1, 3)?;
        let device = sliced.to_device_tensor()?;
        assert!(matches!(device, DeviceTensor::Owned(_)));
        close(&device.to_host()?.into_tensor(), &input.slice(4, 1, 3)?)
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
    let two = model.add_const("two", tensor2(&[[2.0f32]]))?;
    let x2 = model.wire_node("mul", mul(), &[x, two])?[0];
    let y = model.wire_node(
        "attn_sm",
        tract_core::ops::nn::Softmax::new(
            tvec![1],
            None,
            tract_core::ops::nn::SoftmaxKind::LogSoftmax,
        ),
        &[x2],
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

/// An op on the CPU over the model's inputs needs no readback: its result goes up to the
/// device, so the model runs without JSPI.
#[test]
fn coverage_accepts_a_host_prologue() -> TractResult<()> {
    crate::context::wgpu_context();
    let mut model = TypedModel::default();
    let x = model.add_source("x", i64::fact([2, 4]))?;
    let xf = model.wire_node("cast", tract_core::ops::cast::cast(f32::datum_type()), &[x])?[0];
    let two = model.add_const("two", tensor2(&[[2.0f32]]))?;
    let y = model.wire_node("mul", mul(), &[xf, two])?[0];
    model.select_output_outlets(&[y])?;
    WgpuTransform.transform(&mut model)?;
    model = model.into_optimized()?;
    ensure!(!crate::uncovered_ops(&model)?.is_empty(), "the cast should stay on the CPU");
    crate::ensure_wgpu_coverage(&model)
}

/// Squeeze/excitation blocks use full-spatial average pools. The WGPU path
/// reduces each channel across a workgroup instead of making one invocation
/// walk the complete image serially.
#[test]
fn global_average_pool_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{PaddingSpec, PoolSpec, SumPool};
    use tract_core::ops::nn::DataFormat;
    for (n, c, h, w) in [(1usize, 16usize, 64usize, 64usize), (2, 3, 7, 11)] {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([n, c, h, w]))?;
        let pool = SumPool {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![h, w],
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: Some(tvec![h, w]),
                input_channels: c,
                output_channels: c,
            },
            count_include_pad: true,
            normalize: true,
        };
        let y = model.wire_node("global-average", pool, &[x])?[0];
        model.select_output_outlets(&[y])?;
        gpu_vs_cpu(model, Tensor::from_shape(&[n, c, h, w], &fill(17, n * c * h * w))?)
            .with_context(|| format!("n{n} c{c} {h}x{w}"))?;
    }
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
fn axis_op_at_a_model_output_is_left_alone() -> TractResult<()> {
    use tract_gpu::ops::change_axes::GpuAxisOp;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 1, 15360]))?;
    let reshape =
        GpuAxisOp::new(AxisOp::Reshape(2, tvec!(15360.into()), tvec!(96.into(), 160.into())));
    let y = model.wire_node("unmerge", reshape.clone(), &[x])?[0];
    model.select_output_outlets(&[y])?;
    let node = model.node(y.node);
    let patch = crate::rewrite_rules::fuse_axis_op(&(), &model, node, &node.name, &reshape)?;
    ensure!(patch.is_none(), "an axis op with no successor has nothing to fuse into");
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

#[test]
fn rgba8_ingest_applies_normalization() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let rgba: [u8; 8] = [255, 0, 51, 255, 0, 255, 204, 255];
        let normalization = crate::Normalization { mean: [0.5, 0.25, 0.0], std: [0.5, 0.25, 2.0] };
        let gpu = crate::kernels::ingest::tensor_from_rgba8_normalized(2, 1, &rgba, normalization)?;
        q.flush()?;
        let host = gpu.to_host()?.into_tensor();
        let expected = Tensor::from_shape(&[1, 3, 1, 2], &[1.0f32, -1.0, -1.0, 3.0, 0.1, 0.4])?;
        close(&host, &expected)?;
        let bad = crate::Normalization { mean: [0.0; 3], std: [1.0, 0.0, 1.0] };
        ensure!(crate::kernels::ingest::tensor_from_rgba8_normalized(2, 1, &rgba, bad).is_err());
        Ok(())
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

/// A resize over the two trailing axes runs as one launch, two taps unrolled
/// when it is bilinear and the generic gather otherwise, and matches the CPU.
#[test]
fn resize_two_trailing_axes() -> TractResult<()> {
    use tract_core::ops::nn::resize::{CoordTransformer, Interpolator, Nearest, Resize};
    for (interp, coord, shape, scales) in [
        (Interpolator::Linear, CoordTransformer::HalfPixel, [1, 3, 9, 16], [1., 1., 1., 1.]),
        (Interpolator::Linear, CoordTransformer::HalfPixel, [1usize, 3, 9, 16], [1f32, 1., 2., 2.]),
        (Interpolator::Linear, CoordTransformer::AlignCorners, [2, 5, 7, 11], [1., 1., 3., 1.5]),
        (Interpolator::Nearest, CoordTransformer::Asymmetric, [1, 2, 6, 6], [1., 1., 2., 2.]),
        (Interpolator::Cubic, CoordTransformer::HalfPixel, [1, 1, 8, 8], [1., 1., 2., 2.]),
    ] {
        let label = format!("{interp:?} {coord:?} {shape:?} x {scales:?}");
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact(&shape))?;
        let s = model.add_const("scales", tensor1(&scales))?;
        let y = model.wire_node(
            "resize",
            Resize {
                coord_transformer: coord,
                interpolator: interp,
                nearest: Nearest::RoundPreferCeil,
                optional_scales_input: Some(1),
                optional_sizes_input: None,
            },
            &[x, s],
        )?[0];
        model.select_output_outlets(&[y])?;
        let n: usize = shape.iter().product();
        gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(7, n))?).with_context(|| label)?;
    }
    Ok(())
}

/// A dense 3x3 convolution takes the per-pixel kernel, every output channel
/// of a pixel from one thread; the shapes cover the stem, a batch, a stride-1
/// layer with more input channels, and an odd size, against the CPU.
#[test]
fn conv_3x3_per_pixel() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    for (n, ci, co, h, w, stride, pad) in [
        (1usize, 3usize, 16usize, 144usize, 256usize, 2usize, 1usize),
        (2, 3, 8, 20, 30, 2, 1),
        (1, 7, 5, 13, 17, 1, 1),
        (1, 4, 6, 9, 9, 1, 0),
    ] {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([n, ci, h, w]))?;
        let k =
            model.add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &fill(3, co * ci * 9))?)?;
        let b = model.add_const("b", Tensor::from_shape(&[co], &fill(4, co))?)?;
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![3, 3],
                padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, pad]),
                dilations: None,
                strides: Some(tvec![stride, stride]),
                input_channels: ci,
                output_channels: co,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let y = model.wire_node("conv", conv, &[x, k, b])?[0];
        model.select_output_outlets(&[y])?;
        gpu_vs_cpu(model, Tensor::from_shape(&[n, ci, h, w], &fill(5, n * ci * h * w))?)
            .with_context(|| format!("n{n} ci{ci} co{co} {h}x{w} s{stride} p{pad}"))?;
    }
    Ok(())
}

/// The bias and activation after a transposed convolution fold into its
/// kernel, and the result matches the CPU: the segmenter's last layer.
#[test]
fn deconv_bias_activation_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    for (ci, co, h, w) in [(16usize, 1usize, 9usize, 16usize), (3, 2, 5, 7)] {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, ci, h, w]))?;
        let k =
            model.add_const("k", Tensor::from_shape(&[co, ci, 2, 2], &fill(3, ci * co * 4))?)?;
        let bias = model.add_const("bias", Tensor::from_shape(&[co], &fill(4, co))?)?;
        let deconv = Deconv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![2, 2],
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: Some(tvec![2, 2]),
                input_channels: ci,
                output_channels: co,
            },
            kernel_format: KernelFormat::OIHW,
            adjustments: tvec![0, 0],
            group: 1,
        };
        let y = model.wire_node("deconv", deconv, &[x, k, bias])?[0];
        let y = model.wire_node("sigmoid", tract_core::ops::nn::sigmoid(), &[y])?[0];
        model.select_output_outlets(&[y])?;
        let mut folded = model.clone();
        WgpuTransform.transform(&mut folded)?;
        let deconv = folded
            .nodes()
            .iter()
            .find_map(|n| n.op_as::<crate::ops::deconv::WgpuDeconv>())
            .context("no WgpuDeconv after transform")?;
        ensure!(
            deconv.epilogue.len() == 2,
            "epilogue not folded: {:?}\n{}",
            deconv.epilogue,
            folded
                .nodes()
                .iter()
                .map(|n| format!("{} {}", n.op.name(), n.name))
                .collect::<Vec<_>>()
                .join("\n")
        );
        gpu_vs_cpu(model, Tensor::from_shape(&[1, ci, h, w], &fill(5, ci * h * w))?)
            .with_context(|| format!("ci{ci} co{co} {h}x{w}"))?;
    }
    Ok(())
}

/// A residual: the operand is another tensor shaped like the product, read
/// element for element. Two products chained so the residual is an op's
/// output. With `transposed` every flag is set, as a channels-first pointwise
/// convolution lowers, and the sizes are multiples of four so the blocked
/// kernel takes it.
fn matmul_residual(m: usize, k: usize, n: usize, transposed: bool) -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    let mut model = TypedModel::default();
    let op = PrefixMatMul {
        transpose_a: transposed,
        transpose_b: transposed,
        transpose_c: transposed,
        quantize_output: None,
        operating_dt: None,
    };
    let (a_shape, b_shape, bias_shape, w2_shape) = if transposed {
        (vec![k, m], vec![n, k], vec![n, 1], vec![n, n])
    } else {
        (vec![m, k], vec![k, n], vec![1, n], vec![n, n])
    };
    let a = model.add_source("a", f32::fact(&*a_shape))?;
    let b = model.add_const("b", Tensor::from_shape(&b_shape, &fill(1, k * n))?)?;
    let bias = model.add_const("bias", Tensor::from_shape(&bias_shape, &fill(2, n))?)?;
    let first = model.wire_node("first", op, &[a, b])?[0];
    let first = model.wire_node("first_bias", tract_core::ops::math::add(), &[first, bias])?[0];
    let w2 = model.add_const("w2", Tensor::from_shape(&w2_shape, &fill(3, n * n))?)?;
    let second = model.wire_node("second", op, &[first, w2])?[0];
    let y = model.wire_node("residual", tract_core::ops::math::add(), &[second, first])?[0];
    model.select_output_outlets(&[y])?;
    gpu_vs_cpu(model, Tensor::from_shape(&a_shape, &fill(4, m * k))?)
}

/// `k` a multiple of four but not of eight, so the blocked kernel's unrolled
/// loop leaves a four-step tail to run.
#[test]
fn matmul_blocked_k_tail_matches_cpu() -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    for (m, k, n) in [(64usize, 12usize, 16usize), (144, 20, 32), (16, 4, 8)] {
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([k, m]))?;
        let b = model.add_const("b", Tensor::from_shape(&[n, k], &fill(1, n * k))?)?;
        let op = PrefixMatMul {
            transpose_a: true,
            transpose_b: true,
            transpose_c: true,
            quantize_output: None,
            operating_dt: None,
        };
        let out = model.wire_node("mm", op, &[a, b])?[0];
        let bias = model.add_const("bias", Tensor::from_shape(&[n, 1], &fill(2, n))?)?;
        let y = model.wire_node("bias_add", tract_core::ops::math::add(), &[out, bias])?[0];
        model.select_output_outlets(&[y])?;
        gpu_vs_cpu(model, Tensor::from_shape(&[k, m], &fill(3, k * m))?)
            .with_context(|| format!("m={m} k={k} n={n}"))?;
    }
    Ok(())
}

#[test]
fn matmul_residual_matches_cpu() -> TractResult<()> {
    matmul_residual(6, 5, 7, false)?;
    matmul_residual(8, 8, 16, true)
}

#[test]
fn group_norm_runs_on_device() -> TractResult<()> {
    use tract_core::ops::nn::GroupNorm;
    for (shape, axis, reduced_axes) in
        [([2, 32, 9, 7], 1, tvec![2, 3]), ([2, 9, 7, 32], 3, tvec![1, 2])]
    {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact(shape))?;
        let op = GroupNorm { axis, groups: 8, reduced_axes, eps: rctensor0(1e-5f32) };
        let y = model.wire_node("group_norm", op, &[x])?[0];
        model.select_output_outlets(&[y])?;
        crate::ensure_wgpu_coverage(&transformed(model)?)?;
    }
    Ok(())
}

#[test]
fn layer_norm_runs_on_device() -> TractResult<()> {
    use tract_core::ops::math::{add, mul};
    use tract_core::ops::nn::LayerNorm;
    for (axis, affine) in [(2, false), (2, true), (1, false)] {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([2, 77, 48]))?;
        let op = LayerNorm { axis, eps: rctensor0(1e-5f32) };
        let mut y = model.wire_node("layer_norm", op, &[x])?[0];
        if affine {
            let gamma = model.add_const("gamma", Tensor::from_shape(&[1, 1, 48], &fill(1, 48))?)?;
            let beta = model.add_const("beta", Tensor::from_shape(&[1, 1, 48], &fill(2, 48))?)?;
            y = model.wire_node("scale", mul(), &[y, gamma])?[0];
            y = model.wire_node("shift", add(), &[y, beta])?[0];
        }
        model.select_output_outlets(&[y])?;
        let model = transformed(model)?;
        crate::ensure_wgpu_coverage(&model)?;
        let norms = model
            .nodes()
            .iter()
            .filter(|n| n.op_is::<tract_gpu::ops::layer_norm::GpuLayerNorm>())
            .collect::<Vec<_>>();
        ensure!(norms.len() == 1, "expected one device layer norm, got {}", norms.len());
        let inputs = if affine { 3 } else { 1 };
        ensure!(norms[0].inputs.len() == inputs, "gamma and beta should fold into the norm");
    }
    Ok(())
}

#[test]
fn rms_norm_runs_on_device() -> TractResult<()> {
    use tract_transformers::ops::rms_norm::RmsNorm;
    for (axis, scaled) in [(2, false), (2, true), (1, false), (1, true)] {
        let shape = [2, 77, 48];
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact(shape))?;
        let op = RmsNorm { axis, eps: rctensor0(1e-5f32) };
        let mut y = model.wire_node("rms_norm", op, &[x])?[0];
        if scaled {
            let mut gamma_shape = [1, 1, 1];
            gamma_shape[axis] = shape[axis];
            let gamma = Tensor::from_shape(&gamma_shape, &fill(3, shape[axis]))?;
            let gamma = model.add_const("gamma", gamma)?;
            y = model.wire_node("scale", mul(), &[y, gamma])?[0];
        }
        model.select_output_outlets(&[y])?;
        let gpu = transformed(model.clone())?;
        crate::ensure_wgpu_coverage(&gpu)?;
        let norms = gpu
            .nodes()
            .iter()
            .filter_map(|n| n.op_as::<tract_gpu::ops::rms_norm::GpuRmsNorm>())
            .collect::<Vec<_>>();
        ensure!(norms.len() == 1, "expected one device rms norm, got {}", norms.len());
        ensure!(norms[0].has_scale == scaled, "gamma should fold into the norm");
        let len = shape.iter().product();
        gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(5, len))?)
            .with_context(|| format!("axis {axis} scaled {scaled}"))?;
    }
    Ok(())
}

#[test]
fn leaky_relu_matches_cpu() -> TractResult<()> {
    use tract_core::ops::nn::leaky_relu;
    use tract_gpu::ops::leaky_relu::GpuLeakyRelu;
    for (dt, alpha) in
        [(f32::datum_type(), 0.1f32), (f32::datum_type(), 0.01), (f16::datum_type(), 0.2)]
    {
        let len = 10000;
        let mut model = TypedModel::default();
        let x = model.add_source("x", dt.fact([len]))?;
        let y = model.wire_node("act", leaky_relu(alpha), &[x])?[0];
        model.select_output_outlets(&[y])?;
        let gpu = transformed(model.clone())?;
        crate::ensure_wgpu_coverage(&gpu)?;
        ensure!(
            gpu.nodes().iter().any(|n| n.op_is::<GpuLeakyRelu>()),
            "leaky ReLU stayed on the CPU"
        );
        let mut values = fill(9, len);
        values[0] = 0.0;
        let input = Tensor::from_shape(&[len], &values)?.cast_to_dt(dt)?.into_owned();
        gpu_vs_cpu(model, input).with_context(|| format!("{dt:?} alpha {alpha}"))?;
    }
    Ok(())
}

/// Attention scores under a causal `-inf` mask broadcast over batch and
/// heads, with rows both shorter and longer than one workgroup.
#[test]
fn scaled_masked_softmax_runs_on_device() -> TractResult<()> {
    use tract_gpu::ops::scaled_masked_softmax::GpuScaledMaskedSoftmax;
    use tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax;
    for (cols, f16_) in [(7usize, false), (300, false), (300, true)] {
        let rows = 5;
        let shape = [2, 3, rows, cols];
        let causal = (0..rows * cols)
            .map(|i| if i % cols > i / cols + cols - rows { f32::NEG_INFINITY } else { 0.0 })
            .collect::<Vec<_>>();
        let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
        let mut model = TypedModel::default();
        let x = model.add_source("x", dt.fact(shape))?;
        let mask = Tensor::from_shape(&[1, 1, rows, cols], &causal)?.cast_to_dt(dt)?.into_owned();
        let mask = model.add_const("mask", mask)?;
        let op = ScaledMaskedSoftmax { scale: rctensor0(0.125f32), post_softmax_mask: false };
        let y = model.wire_node("softmax", op, &[x, mask])?[0];
        model.select_output_outlets(&[y])?;
        let gpu = transformed(model.clone())?;
        crate::ensure_wgpu_coverage(&gpu)?;
        ensure!(
            gpu.nodes().iter().any(|n| n.op_is::<GpuScaledMaskedSoftmax>()),
            "masked softmax stayed on the CPU"
        );
        let input = Tensor::from_shape(&shape, &fill(6, shape.iter().product()))?;
        gpu_vs_cpu(model, input.cast_to_dt(dt)?.into_owned())
            .with_context(|| format!("cols {cols} f16 {f16_}"))?;
    }
    Ok(())
}

/// Products the tiled matmul takes: every dimension off the tile size, each
/// transposition, a prefix, f16, a bias epilogue, and thin products with a long
/// `k` that fill one or two tiles.
#[test]
fn matmul_tiled_matches_cpu() -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    for (prefix, m, k, n, ta, tb, tc, f16_, bias) in [
        (1usize, 130usize, 17usize, 513usize, false, false, false, false, false),
        (1, 130, 17, 513, true, false, false, false, false),
        (1, 130, 17, 513, false, true, false, false, false),
        (1, 130, 17, 513, false, false, true, false, false),
        (1, 130, 17, 513, true, true, true, false, false),
        (8, 65, 33, 63, false, false, false, false, false),
        (1, 130, 17, 513, false, false, false, true, false),
        (1, 130, 17, 513, false, false, false, false, true),
        (1, 16, 300, 16, false, false, false, false, false),
        (1, 20, 960, 40, true, false, false, false, false),
    ] {
        let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
        let a_shape = [prefix, if ta { k } else { m }, if ta { m } else { k }];
        let b_shape = [prefix, if tb { n } else { k }, if tb { k } else { n }];
        let mut model = TypedModel::default();
        let a = model.add_source("a", dt.fact(a_shape))?;
        let b_data = Tensor::from_shape(&b_shape, &fill(1, b_shape.iter().product()))?;
        let b = model.add_const("b", b_data.cast_to_dt(dt)?.into_owned())?;
        let op = PrefixMatMul {
            transpose_a: ta,
            transpose_b: tb,
            transpose_c: tc,
            quantize_output: None,
            operating_dt: None,
        };
        let mut y = model.wire_node("mm", op, &[a, b])?[0];
        if bias {
            let bias = Tensor::from_shape(&[1, 1, n], &fill(2, n))?;
            let bias = model.add_const("bias", bias.cast_to_dt(dt)?.into_owned())?;
            y = model.wire_node("bias_add", tract_core::ops::math::add(), &[y, bias])?[0];
        }
        model.select_output_outlets(&[y])?;
        let input = Tensor::from_shape(&a_shape, &fill(3, a_shape.iter().product()))?;
        gpu_vs_cpu(model, input.cast_to_dt(dt)?.into_owned()).with_context(|| {
            format!("prefix={prefix} m={m} k={k} n={n} ta={ta} tb={tb} tc={tc} f16={f16_}")
        })?;
    }
    Ok(())
}

/// Convolutions the implicit-GEMM kernel takes: strides, dilation, groups, a
/// batch, channels-last, an uneven kernel and padding, f16, a bias, whole
/// tiles, and a long reduction over a small map that fills only a few tiles.
#[test]
fn conv_tiled_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    #[rustfmt::skip]
    let cases: [(DataFormat, usize, usize, usize, [usize; 2], [usize; 2], [usize; 2], [usize; 2], usize, bool); 11] = [
        // fmt, n, ci, co, hw, kernel, stride, dilation, pad, f16
        (DataFormat::NCHW, 1, 8, 40, [46, 46], [3, 3], [1, 1], [1, 1], 1, false),
        (DataFormat::NCHW, 1, 8, 40, [90, 90], [3, 3], [2, 2], [1, 1], 1, false),
        (DataFormat::NCHW, 1, 8, 40, [49, 49], [3, 3], [1, 1], [2, 2], 0, false),
        (DataFormat::NCHW, 1, 16, 80, [33, 33], [3, 3], [1, 1], [1, 1], 1, false),
        (DataFormat::NCHW, 2, 8, 40, [33, 33], [3, 3], [1, 1], [1, 1], 1, false),
        (DataFormat::NHWC, 1, 8, 40, [46, 46], [3, 3], [1, 1], [1, 1], 1, false),
        (DataFormat::NCHW, 1, 8, 40, [45, 47], [1, 5], [1, 1], [1, 1], 2, false),
        (DataFormat::NCHW, 1, 8, 40, [46, 46], [3, 3], [1, 1], [1, 1], 1, true),
        (DataFormat::NCHW, 1, 32, 64, [46, 46], [3, 3], [1, 1], [1, 1], 1, false),
        (DataFormat::NCHW, 1, 32, 64, [46, 46], [3, 3], [1, 1], [1, 1], 1, true),
        (DataFormat::NCHW, 1, 64, 40, [7, 12], [5, 5], [1, 1], [1, 1], 2, false),
    ];
    for (i, (fmt, n, ci, co, hw, kernel, stride, dilation, pad, f16_)) in
        cases.into_iter().enumerate()
    {
        let group = if ci == 16 { 2 } else { 1 };
        let shape = fmt.from_n_c_hw(n, ci, hw)?.shape.to_vec();
        let k_shape = [co, ci / group, kernel[0], kernel[1]];
        // An f16 case runs on the GPU in f16 and is checked against the CPU in f32 on the
        // same f16-rounded values: the CPU's own f16 convolution is the less exact of the two.
        let values = |seed, shape: &[usize]| -> TractResult<Tensor> {
            let t = Tensor::from_shape(shape, &fill(seed, shape.iter().product()))?;
            Ok(if f16_ { t.cast_to::<f16>()?.cast_to::<f32>()?.into_owned() } else { t })
        };
        let input = values(3, &shape)?;
        let build = |dt: DatumType| -> TractResult<TypedModel> {
            let mut model = TypedModel::default();
            let x = model.add_source("x", dt.fact(&shape))?;
            let k = model.add_const("k", values(1, &k_shape)?.cast_to_dt(dt)?.into_owned())?;
            let b = model.add_const("b", values(2, &[co])?.cast_to_dt(dt)?.into_owned())?;
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: fmt,
                    kernel_shape: kernel.iter().copied().collect(),
                    padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, 0]),
                    dilations: Some(dilation.iter().copied().collect()),
                    strides: Some(stride.iter().copied().collect()),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            Ok(model)
        };
        if f16_ {
            let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
            let gpu = rt.prepare(build(f16::datum_type())?)?;
            let g = gpu.run(tvec!(input.cast_to::<f16>()?.into_owned().into()))?.remove(0);
            let cpu = Arc::new(SimplePlan::new(build(f32::datum_type())?)?);
            let c = cpu.run(tvec!(input.clone().into()))?.remove(0);
            let c = c.cast_to::<f16>()?.into_owned();
            g.into_tensor().close_enough(&c, Approximation::Approximate)
        } else {
            gpu_vs_cpu(build(f32::datum_type())?, input)
        }
        .with_context(|| format!("case {i}"))?;
    }
    Ok(())
}

/// `Iff(IsNan(x), value, x)` runs on the device as one op, NaNs replaced and other values
/// untouched, with no boolean tensor left on the CPU.
#[test]
fn nan_replace_runs_on_device() -> TractResult<()> {
    use tract_core::ops::logic::Iff;
    use tract_core::ops::math::is_nan;
    for (dt, value) in [(f32::datum_type(), 0.0f32), (f16::datum_type(), -2.5)] {
        let build = || -> TractResult<TypedModel> {
            let mut model = TypedModel::default();
            let x = model.add_source("x", dt.fact([3, 5]))?;
            let nan = model.wire_node("nan", is_nan(), &[x])?[0];
            let value = tensor2(&[[value]]).cast_to_dt(dt)?.into_owned();
            let value = model.add_const("value", value)?;
            let y = model.wire_node("y", Iff, &[nan, value, x])?[0];
            model.select_output_outlets(&[y])?;
            Ok(model)
        };
        crate::ensure_wgpu_coverage(&transformed(build()?)?)?;
        let mut data = fill(1, 15);
        for i in [0, 6, 14] {
            data[i] = f32::NAN;
        }
        let input = Tensor::from_shape(&[3, 5], &data)?.cast_to_dt(dt)?.into_owned();
        gpu_vs_cpu(build()?, input).with_context(|| format!("{dt:?}"))?;
    }
    Ok(())
}

/// An op left on the CPU reads its constant inputs from the host: there is no readback of the
/// copy uploaded for device consumers, so a token lookup over an embedding table runs without
/// JSPI.
#[test]
fn cpu_op_reads_constants_from_the_host() -> TractResult<()> {
    use tract_core::ops::array::Gather;
    let mut model = TypedModel::default();
    let ids = model.add_source("ids", i64::fact([2, 3]))?;
    let table = model.add_const(
        "table",
        Tensor::from_shape(&[10, 4], &fill(1, 40))?.cast_to::<f16>()?.into_owned(),
    )?;
    let embedded = model.wire_node("embedding", Gather::new(0), &[table, ids])?[0];
    let two = model.add_const("two", tensor3(&[[[f16::from_f32(2.0)]]]))?;
    let y = model.wire_node("mul", mul(), &[embedded, two])?[0];
    model.select_output_outlets(&[y])?;
    // Later passes rewire such a readback away, but only after it was evaluated, which a browser
    // cannot do: check the translation itself.
    use tract_core::internal::translator::Translate;
    crate::context::wgpu_context();
    let translated = WgpuTransform.translate_model(&model)?;
    let readbacks = crate::mid_graph_readbacks(&translated)?;
    ensure!(readbacks.is_empty(), "unexpected readbacks: {readbacks:?}");
    gpu_vs_cpu(model, tensor2(&[[0i64, 3, 9], [1, 1, 7]]))
}

/// Launches of more workgroups than one grid dimension holds, which `dispatch`
/// folds into a second dimension: a matmul and a convolution over 4M outputs.
#[test]
fn launches_past_one_grid_dimension_match_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::nn::DataFormat;
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact([2100, 1]))?;
    let b = model.add_const("b", Tensor::from_shape(&[1, 2100], &fill(1, 2100))?)?;
    let op = PrefixMatMul {
        transpose_a: false,
        transpose_b: false,
        transpose_c: false,
        quantize_output: None,
        operating_dt: None,
    };
    let y = model.wire_node("mm", op, &[a, b])?[0];
    model.select_output_outlets(&[y])?;
    gpu_vs_cpu(model, Tensor::from_shape(&[2100, 1], &fill(2, 2100))?).context("matmul")?;

    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 1, 1000, 1000]))?;
    let k = model.add_const("k", Tensor::from_shape(&[5, 1, 1, 3], &fill(3, 15))?)?;
    let bias = model.add_const("bias", rctensor0(0f32))?;
    let conv = Conv {
        pool_spec: PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec![1, 3],
            padding: PaddingSpec::Valid,
            dilations: None,
            strides: None,
            input_channels: 1,
            output_channels: 5,
        },
        kernel_fmt: KernelFormat::OIHW,
        group: 1,
        q_params: None,
    };
    let y = model.wire_node("conv", conv, &[x, k, bias])?[0];
    model.select_output_outlets(&[y])?;
    gpu_vs_cpu(model, Tensor::from_shape(&[1, 1, 1000, 1000], &fill(4, 1_000_000))?).context("conv")
}

/// Runs the model `build` makes in f16 on the GPU and in f32 on the CPU.
fn f16_vs_f32(
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

/// f16 kernels keep f32 intermediates: a sigmoid deep in its tail, and a
/// softmax and a sum over thousands of terms.
#[test]
fn f16_keeps_f32_intermediates() -> TractResult<()> {
    use tract_core::ops::nn::{Reduce, Reducer, Softmax, SoftmaxKind, sigmoid};
    crate::context::wgpu_context();
    let tail: Vec<f32> = fill(3, 1024).iter().map(|x| x * 3.0 - 3.0).collect();
    let sigmoid_model = |dt: DatumType| {
        let mut model = TypedModel::default();
        let x = model.add_source("x", dt.fact([1024]))?;
        let y = model.wire_node("sigmoid", sigmoid(), &[x])?[0];
        model.select_output_outlets(&[y])?;
        Ok(model)
    };
    f16_vs_f32(sigmoid_model, tensor1(&tail), Approximation::Custom(1e-6, 2e-3, 0.0))
        .context("sigmoid")?;
    let long = Tensor::from_shape(&[2, 4096], &fill(4, 2 * 4096))?;
    let softmax_model = |dt: DatumType| {
        let mut model = TypedModel::default();
        let x = model.add_source("x", dt.fact([2, 4096]))?;
        let op = Softmax { axes: tvec![1], quant_output_dt: None, kind: SoftmaxKind::Softmax };
        let y = model.wire_node("softmax", op, &[x])?[0];
        model.select_output_outlets(&[y])?;
        Ok(model)
    };
    f16_vs_f32(softmax_model, long.clone(), Approximation::Custom(1e-7, 2e-3, 0.0))
        .context("softmax")?;
    let sum_model = |dt: DatumType| {
        let mut model = TypedModel::default();
        let x = model.add_source("x", dt.fact([2, 4096]))?;
        let y = model.wire_node("sum", Reduce { axes: tvec![1], reducer: Reducer::Sum }, &[x])?[0];
        model.select_output_outlets(&[y])?;
        Ok(model)
    };
    f16_vs_f32(sum_model, long, Approximation::Custom(0.5, 2e-3, 0.0)).context("sum")
}

/// Attention shapes: queries and keys off the tile sizes, rank 3 and 4, a value
/// head dim unlike the query's, the largest head dim the flash kernel takes, f16
/// and an explicit scale; a head dim past it falls back to primitives, still all
/// on the device.
#[test]
fn flash_sdpa_matches_cpu() -> TractResult<()> {
    use tract_transformers::ops::sdpa::Sdpa;
    #[rustfmt::skip]
    let cases: [(&[usize], usize, usize, usize, usize, bool, Option<f32>, bool); 6] = [
        // prefix, sq, sk, d, dv, f16, scale, flash
        (&[2, 3], 70, 77, 40, 40, false, None, true),
        (&[2], 33, 128, 16, 24, false, Some(0.3), true),
        (&[1, 2], 64, 64, 48, 48, false, None, true),
        (&[2, 3], 70, 77, 40, 40, true, None, true),
        (&[1, 2], 40, 200, 8, 8, true, None, true),
        (&[1, 2], 40, 50, 64, 64, false, None, false),
    ];
    for (i, (prefix, sq, sk, d, dv, f16_, scale, flash)) in cases.into_iter().enumerate() {
        let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
        let shape = |s: usize, c: usize| prefix.iter().copied().chain([s, c]).collect::<Vec<_>>();
        let values = |seed, shape: &[usize]| -> TractResult<Tensor> {
            let centred: Vec<f32> =
                fill(seed, shape.iter().product()).iter().map(|x| x + 1.5).collect();
            Ok(Tensor::from_shape(shape, &centred)?.cast_to_dt(dt)?.into_owned())
        };
        let inputs =
            [values(1, &shape(sq, d))?, values(2, &shape(sk, d))?, values(3, &shape(sk, dv))?];
        let mut model = TypedModel::default();
        let wires = ["q", "k", "v"]
            .iter()
            .zip(&inputs)
            .map(|(name, t)| model.add_source(*name, dt.fact(t.shape())))
            .collect::<TractResult<Vec<_>>>()?;
        let op = Sdpa {
            scale: scale.map(tensor0),
            datum_type: dt,
            acc_datum_type: f32::datum_type(),
            is_causal: false,
        };
        let y = model.wire_node("sdpa", op, &wires)?[0];
        model.select_output_outlets(&[y])?;
        let translated = transformed(model.clone())?;
        crate::ensure_wgpu_coverage(&translated).with_context(|| format!("case {i}"))?;
        let flashed =
            translated.nodes().iter().any(|n| n.op_is::<crate::ops::sdpa::WgpuFlashSdpa>());
        ensure!(flashed == flash, "case {i}: flash kernel used: {flashed}");
        let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
        let tv = || inputs.iter().map(|t| t.clone().into_tvalue()).collect::<TVec<_>>();
        let g = rt.prepare(model.clone())?.run(tv())?.remove(0).into_tensor();
        let c = Arc::new(SimplePlan::new(model)?).run(tv())?.remove(0).into_tensor();
        g.close_enough(&c, Approximation::Approximate).with_context(|| format!("case {i}"))?;
    }
    Ok(())
}

/// A bind group keeps the buffers it binds alive, so once a tensor is dropped the groups that
/// bound it must leave the cache rather than hold its memory until the cache fills.
#[test]
fn dropped_buffers_leave_the_bind_group_cache() -> TractResult<()> {
    use tract_core::ops::nn::sigmoid;
    with_wgpu_queue(|q| {
        let input = Tensor::from_shape(&[256], &fill(3, 256))?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F32, input.shape())?;
        let id = crate::utils::get_wgpu_buffer(&input).id;
        wgpu_element_wise_dispatch(&*sigmoid().0, &input, &output)?;
        q.flush()?;
        ensure!(q.context().binds_buffer(id), "the group was not cached");
        drop(input);
        q.flush()?;
        ensure!(!q.context().binds_buffer(id), "the group outlived its buffer");
        Ok(())
    })
}

/// Past its buffer budget, a run submits what it recorded and goes on reusing the buffers it no
/// longer holds, with the same result as a run submitted once at the end. The model is symbolic,
/// so its buffers come from the pool rather than a memory plan.
#[test]
fn frame_budget_submits_early() -> TractResult<()> {
    use tract_core::knobs::KnobScope;
    use tract_core::ops::math::{add, mul};
    crate::context::wgpu_context();
    let shape = [1usize, 16, 128, 128];
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let mut x = model.add_source("x", f32::fact(dims![1, 16, s, 128]))?;
    let source = x;
    for i in 0..8 {
        let y = model.wire_node(format!("mul{i}"), mul(), &[x, source])?[0];
        x = model.wire_node(format!("add{i}"), add(), &[y, source])?[0];
    }
    model.select_output_outlets(&[x])?;
    let n: usize = shape.iter().product();
    let input =
        Tensor::from_shape(&shape, &fill(9, n).iter().map(|v| v * 0.1).collect::<Vec<_>>())?;
    let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
    let cpu = Arc::new(SimplePlan::new(model.clone())?).run(tvec!(input.clone().into()))?.remove(0);
    for (budget, early) in [("1", true), ("0", false)] {
        KnobScope::new([("TRACT_WGPU_FRAME_BUDGET_MB", budget)])?.enter(
            || -> TractResult<()> {
                let gpu = rt.prepare(model.clone())?;
                let before = with_wgpu_queue(|q| Ok(q.frames_submitted()))?;
                let out = gpu.run(tvec!(input.clone().into()))?.remove(0).into_tensor();
                let frames = with_wgpu_queue(|q| Ok(q.frames_submitted()))? - before;
                ensure!((frames > 1) == early, "budget {budget}: {frames} submissions");
                out.close_enough(&cpu, Approximation::Close)
            },
        )?;
    }
    Ok(())
}

/// Convolutions over one spatial axis run as two-dimensional ones on the device: dilated,
/// padded, strided, channels-last.
#[test]
fn conv_1d_runs_on_device() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    #[rustfmt::skip]
    let cases = [
        // fmt, ci, co, len, kernel, stride, dilation, pad
        (DataFormat::NCHW, 8, 16, 50, 3, 1, 2, 2),
        (DataFormat::NCHW, 16, 40, 200, 2, 1, 4, 0),
        (DataFormat::NHWC, 8, 16, 50, 3, 2, 1, 1),
    ];
    for (i, (fmt, ci, co, len, kernel, stride, dilation, pad)) in cases.into_iter().enumerate() {
        let shape = fmt.from_n_c_hw(1, ci, [len])?.shape.to_vec();
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact(&shape))?;
        let k = model
            .add_const("k", Tensor::from_shape(&[co, ci, kernel], &fill(1, co * ci * kernel))?)?;
        let b = model.add_const("b", Tensor::from_shape(&[co], &fill(2, co))?)?;
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: fmt,
                kernel_shape: tvec![kernel],
                padding: PaddingSpec::Explicit(tvec![pad], tvec![pad]),
                dilations: Some(tvec![dilation]),
                strides: Some(tvec![stride]),
                input_channels: ci,
                output_channels: co,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let y = model.wire_node("conv", conv, &[x, k, b])?[0];
        model.select_output_outlets(&[y])?;
        let translated = transformed(model.clone())?;
        ensure!(
            translated.nodes().iter().any(|n| n.op.name() == "WgpuConv"),
            "case {i}: the convolution did not reach the device"
        );
        let n: usize = shape.iter().product();
        gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(3, n))?)
            .with_context(|| format!("case {i}"))?;
    }
    Ok(())
}

/// Gathers `[2, 6, 5, 3]` data at coordinates computed from `x` like a deformable convolution's
/// sampling corners: rounded with `round` if given, cast to integers, moved by one, clipped into
/// range and stacked.
fn sampling_gather(round: Option<Box<dyn ElementWiseMiniOp>>) -> TractResult<TypedModel> {
    use tract_core::ops::array::{GatherNd, TypedConcat};
    use tract_core::ops::cast::cast;
    use tract_core::ops::change_axes::AxisOp;
    use tract_core::ops::element_wise::ElementWiseOp;
    use tract_core::ops::math::{add, max, min};
    let mut model = TypedModel::default();
    let mut wire = model.add_source("x", f32::fact([2, 4, 2]))?;
    if let Some(round) = round {
        wire = model.wire_node("round", ElementWiseOp(round, None), &[wire])?[0];
    }
    wire = model.wire_node("cast", cast(i64::datum_type()), &[wire])?[0];
    let one = model.add_const("one", tensor3(&[[[1i64]]]))?;
    let low = model.add_const("low", tensor3(&[[[0i64]]]))?;
    let high = model.add_const("high", tensor3(&[[[5i64, 4]]]))?;
    wire = model.wire_node("add", add(), &[wire, one])?[0];
    wire = model.wire_node("clip.min", max(), &[wire, low])?[0];
    wire = model.wire_node("clip.max", min(), &[wire, high])?[0];
    let moved = model.wire_node("add_axis", AxisOp::Add(1), &[wire])?[0];
    let moved = model.wire_node("rm_axis", AxisOp::Rm(1), &[moved])?[0];
    let indices = model.wire_node("concat", TypedConcat::new(1), &[wire, moved])?[0];
    let data = Tensor::from_shape(&[2, 6, 5, 3], &fill(7, 2 * 6 * 5 * 3))?;
    let data = model.add_const("data", data)?;
    let y = model.wire_node("gather", GatherNd::new(1), &[data, indices])?[0];
    model.select_output_outlets(&[y])?;
    Ok(model)
}

fn op_names(model: &TypedModel) -> Vec<String> {
    model.nodes().iter().map(|n| n.op.name().to_string()).collect()
}

/// Rounded coordinates moved by integer sums and clips are computed in f32, and the gather and
/// everything feeding it run on the device.
#[test]
fn gather_nd_of_rounded_coordinates_runs_on_device() -> TractResult<()> {
    let model = sampling_gather(Some(Box::new(tract_core::ops::math::Floor {})))?;
    let names = op_names(&transformed(model.clone())?);
    ensure!(names.iter().any(|n| n == "WgpuGatherNd"), "{names:?}");
    ensure!(!names.iter().any(|n| n == "GatherNd" || n == "Cast"), "{names:?}");
    let input =
        Tensor::from_shape(&[2, 4, 2], &fill(8, 16).iter().map(|v| v * 3.).collect::<Vec<_>>())?;
    gpu_vs_cpu(model, input)
}

/// A cast that truncates rather than rounds keeps the gather on the host.
#[test]
fn gather_nd_of_truncated_coordinates_stays_on_host() -> TractResult<()> {
    let model = sampling_gather(None)?;
    let names = op_names(&transformed(model.clone())?);
    ensure!(names.iter().any(|n| n == "GatherNd"), "{names:?}");
    let input =
        Tensor::from_shape(&[2, 4, 2], &fill(9, 16).iter().map(|v| v * 3.).collect::<Vec<_>>())?;
    gpu_vs_cpu(model, input)
}

/// Eight multiply-adds against the source: more intermediates than the plan needs partitions.
fn residual_chain(shape: &[usize]) -> TractResult<TypedModel> {
    use tract_core::ops::math::{add, mul};
    let mut model = TypedModel::default();
    let mut x = model.add_source("x", f32::fact(shape))?;
    let source = x;
    for i in 0..8 {
        let y = model.wire_node(format!("mul{i}"), mul(), &[x, source])?[0];
        x = model.wire_node(format!("add{i}"), add(), &[y, source])?[0];
    }
    model.select_output_outlets(&[x])?;
    Ok(model)
}

/// Runs `model` on the wgpu runtime once per input, checking each against the CPU while every
/// earlier output is still held, and returns how many partition buffers the thread holds after
/// each run.
fn planned_runs(model: TypedModel, inputs: &[Tensor]) -> TractResult<Vec<usize>> {
    crate::context::wgpu_context();
    let cpu = Arc::new(SimplePlan::new(model.clone())?);
    let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
    let gpu = rt.prepare(model)?;
    let mut held = vec![];
    let mut kept = vec![];
    for input in inputs {
        let out = gpu.run(tvec!(input.clone().into()))?.remove(0).into_tensor();
        held.push(with_wgpu_queue(|q| Ok(q.partition_buffers_held()))?);
        kept.push((out, cpu.run(tvec!(input.clone().into()))?.remove(0).into_tensor()));
        for (gpu, cpu) in &kept {
            gpu.close_enough(cpu, Approximation::Approximate)?;
        }
    }
    Ok(held)
}

/// A node's inputs are alive while it runs, so no planned output shares a buffer with an input
/// of the node that writes it: a dispatch never binds one buffer as a read and a write.
#[test]
fn planned_partitions_never_alias_a_dispatch() -> TractResult<()> {
    use tract_gpu::memory::{DeviceMemSchema, SchemaOptions};
    let model = transformed(residual_chain(&[1, 4, 32, 32])?)?;
    let plan = SimplePlan::new(model)?;
    let options =
        SchemaOptions { plan_host_synced: true, exclude_outputs: true, ..Default::default() };
    let schema = DeviceMemSchema::build_with(
        plan.model(),
        plan.order_without_consts(),
        &SymbolValues::default(),
        options,
    )?;
    let resolved = schema.resolve_partitions(&SymbolValues::default())?;
    let partitions = |outlet: OutletId| -> Vec<usize> {
        resolved.partitions_by_node[outlet.node]
            .as_ref()
            .and_then(|slots| slots.get(outlet.slot))
            .map(|it| it.to_vec())
            .unwrap_or_default()
    };
    let mut planned = 0;
    for node in plan.model().nodes() {
        for slot in 0..node.outputs.len() {
            let written = partitions(OutletId::new(node.id, slot));
            planned += written.len();
            for input in &node.inputs {
                let read = partitions(*input);
                ensure!(
                    written.iter().all(|p| !read.contains(p)),
                    "{} writes partitions {written:?} it reads",
                    node.name
                );
            }
        }
    }
    ensure!(planned > resolved.sizes.len(), "{planned} outputs, {:?}", resolved.sizes);
    Ok(())
}

/// A planned model takes its partition buffers back on its next run, while an output of the
/// run before it is still held.
#[test]
fn planned_model_reuses_its_buffers() -> TractResult<()> {
    let shape = [1usize, 4, 32, 32];
    let n = shape.iter().product();
    let inputs = [1, 2, 3].map(|seed| Tensor::from_shape(&shape, &fill(seed, n)).unwrap());
    let held = planned_runs(residual_chain(&shape)?, &inputs)?;
    ensure!(held[0] > 0 && held.iter().all(|h| *h == held[0]), "{held:?}");
    Ok(())
}

/// A matmul whose `b` is computed in the turn reads what this turn wrote to the partition, not
/// a repack of what an earlier turn left there.
#[test]
fn planned_matmul_of_two_activations_runs_twice() -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::math::{add, mul};
    let op = PrefixMatMul {
        transpose_a: false,
        transpose_b: false,
        transpose_c: false,
        quantize_output: None,
        operating_dt: None,
    };
    let shape = [64usize, 64];
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact(shape))?;
    let a = model.wire_node("a", add(), &[x, x])?[0];
    let b = model.wire_node("b", mul(), &[x, x])?[0];
    let y = model.wire_node("y", op, &[a, b])?[0];
    model.select_output_outlets(&[y])?;
    let inputs = [4, 5].map(|seed| Tensor::from_shape(&shape, &fill(seed, 64 * 64)).unwrap());
    planned_runs(model, &inputs)?;
    Ok(())
}

#[test]
fn packed_ingest_rejects_invalid_input() {
    for (width, height, len) in [(0, 2, 0), (2, 0, 0), (2, 2, 7), (2, 2, 9), (u32::MAX, 2, 0)] {
        assert!(crate::tensor_from_rgb565(width, height, &vec![0; len]).is_err());
    }
    for (width, height, len) in
        [(0, 2, 0), (3, 2, 9), (2, 3, 9), (2, 2, 5), (2, 2, 7), (65536, 65536, 0)]
    {
        assert!(crate::tensor_from_nv12(width, height, &vec![0; len]).is_err());
    }
}

#[test]
fn rgb565_ingest_unaligned_slice_and_partial_word() -> TractResult<()> {
    let bytes = [99u8, 0x00, 0xf8, 0xe0, 0x07, 0x1f, 0x00];
    let tensor = crate::tensor_from_rgb565(3, 1, &bytes[1..])?.to_host()?;
    assert_eq!(tensor.shape(), &[1, 3, 1, 3]);
    assert_eq!(
        tensor.try_as_plain_ram()?.as_slice::<f32>()?,
        &[1., 0., 0., 0., 1., 0., 0., 0., 1.]
    );
    Ok(())
}

#[test]
fn rgb565_ingest_all_bit_patterns() -> TractResult<()> {
    let bytes: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
    let tensor = crate::tensor_from_rgb565(256, 256, &bytes)?.to_host()?;
    let values = tensor.try_as_plain_ram()?.as_slice::<f32>()?;
    for pixel in 0..=u16::MAX {
        let expected = [
            f32::from(pixel >> 11) / 31.,
            f32::from((pixel >> 5) & 63) / 63.,
            f32::from(pixel & 31) / 31.,
        ];
        for c in 0..3 {
            assert!((values[c * 65536 + usize::from(pixel)] - expected[c]).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn nv12_ingest_neutral_and_colored_blocks() -> TractResult<()> {
    let bytes = [0u8, 255, 128, 64, 32, 200, 40, 230, 128, 128, 90, 240];
    let tensor = crate::tensor_from_nv12(4, 2, &bytes)?.to_host()?;
    let actual = tensor.try_as_plain_ram()?.as_slice::<f32>()?;
    assert_eq!(tensor.shape(), &[1, 3, 2, 4]);
    let (kr, kb) = (0.299f64, 0.114f64);
    for i in 0..8 {
        let uv = 8 + ((i % 4) / 2) * 2;
        let y = f64::from(bytes[i]);
        let cb = f64::from(bytes[uv]) - 128.;
        let cr = f64::from(bytes[uv + 1]) - 128.;
        let r = y + 2. * (1. - kr) * cr;
        let b = y + 2. * (1. - kb) * cb;
        let g = (y - kr * r - kb * b) / (1. - kr - kb);
        for (c, value) in [r, g, b].into_iter().enumerate() {
            assert!((actual[c * 8 + i] - (value / 255.).clamp(0., 1.) as f32).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn ingested_tensor_survives_to_device_without_a_copy() -> TractResult<()> {
    use tract_gpu::tensor::DeviceTensorExt;
    let input = crate::tensor_from_rgb565(1, 1, &[0, 0xf8])?;
    let ptr = input.device_buffer_ptr();
    let sync = tract_gpu::sync::DeviceSync::new(tract_gpu::sync::DeviceSyncKind::ToDevice);
    let outputs = sync.eval_out_of_plan(tvec![input.into_tensor().into_tvalue()])?.unwrap();
    assert_eq!(outputs[0].to_device_tensor()?.device_buffer_ptr(), ptr);
    Ok(())
}

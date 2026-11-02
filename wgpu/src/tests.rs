#![cfg(test)]

use tract_core::internal::*;
use tract_core::ops::math::mul;
use tract_core::transform::ModelTransform;
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt, IntoDevice};

use crate::kernels::copy::wgpu_copy_nd_dispatch;
use crate::kernels::element_wise::wgpu_element_wise_dispatch;
use crate::test_utils::{close, f16_vs_f32, fill, gpu_vs_cpu, transformed};
use crate::{WgpuTransform, with_wgpu_queue};

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

/// A unit axis removed between two moves keeps the first move's strides, so
/// the whole run rides the add as one view.
#[test]
fn a_move_a_unit_axis_removal_and_a_move_ride_the_op_that_reads_them() -> TractResult<()> {
    use tract_core::ops::math::add;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 3, 2, 2]))?;
    let mut w = model.wire_node("mv0", AxisOp::Move(3, 1), &[x])?[0];
    w = model.wire_node("rm", AxisOp::Rm(0), &[w])?[0];
    w = model.wire_node("mv1", AxisOp::Move(2, 1), &[w])?[0];
    let c = model.add_const("c", Tensor::from_shape(&[2, 2, 3], &fill(7, 12))?)?;
    let y = model.wire_node("add", add(), &[w, c])?[0];
    model.select_output_outlets(&[y])?;
    gpu_vs_cpu(model, Tensor::from_shape(&[1, 3, 2, 2], &fill(3, 12))?)
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

// --- fusion arithmetic: the fused graph must compute what the ops did apart ---

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

/// A `Move` in front of a convolution is copied first: the convolution kernels
/// read their input in its data format's natural layout, not through strides.
#[test]
fn a_move_into_a_convolution_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    for (ci, co, hw) in [(8usize, 40usize, [9usize, 11usize]), (3, 4, [6, 7])] {
        let shape = [1, hw[0], hw[1], ci];
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact(shape))?;
        let m = model.wire_node("mv", AxisOp::Move(3, 1), &[x])?[0];
        let k =
            model.add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &fill(1, co * ci * 9))?)?;
        let b = model.add_const("b", Tensor::from_shape(&[co], &fill(2, co))?)?;
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![3, 3],
                padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                dilations: None,
                strides: None,
                input_channels: ci,
                output_channels: co,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        let y = model.wire_node("conv", conv, &[m, k, b])?[0];
        model.select_output_outlets(&[y])?;
        let input = Tensor::from_shape(&shape, &fill(3, shape.iter().product()))?;
        gpu_vs_cpu(model, input).with_context(|| format!("ci={ci} co={co}"))?;
    }
    Ok(())
}

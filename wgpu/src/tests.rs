#![cfg(test)]

use tract_core::internal::*;
use tract_core::ops::math::mul;
use tract_core::ops::nn::sigmoid;
use tract_core::transform::ModelTransform;
use tract_gpu::tensor::{DeviceTensor, IntoDevice};

use crate::kernels::bin_ops::wgpu_bin_op_dispatch;
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
fn element_wise_sigmoid_matches_cpu() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let data: Vec<f32> = (-8..8).map(|i| i as f32 * 0.25).collect();
        let t = Tensor::from_shape(&[4, 4], &data)?;
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

#[test]
fn binary_mul_broadcast_matches_cpu() -> TractResult<()> {
    with_wgpu_queue(|q| {
        let a = Tensor::from_shape(&[2, 4], &(0..8).map(|i| i as f32 + 1.0).collect::<Vec<_>>())?;
        let b = Tensor::from_shape(&[1, 4], &[1.0f32, 2.0, 3.0, 4.0])?;
        let cpu = mul()
            .eval_out_of_plan(tvec!(a.clone().into_tvalue(), b.clone().into_tvalue()))?
            .unwrap()
            .remove(0)
            .into_tensor();
        let da = a.into_device()?;
        let db = b.into_device()?;
        let out_shape = tract_core::broadcast::multi_broadcast(&[da.shape(), db.shape()])?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F32, &out_shape)?;
        wgpu_bin_op_dispatch(&*mul().0, &da, &db, &output)?;
        q.flush()?;
        close(&output.to_host()?.into_tensor(), &cpu)
    })
}

#[test]
fn two_op_model_mul_then_sigmoid() -> TractResult<()> {
    crate::context::wgpu_context();
    let mut gpu_model = TypedModel::default();
    let x = gpu_model.add_source("x", f32::fact([2, 4]))?;
    let two = gpu_model.add_const("two", tensor2(&[[2.0f32]]))?;
    let y = gpu_model.wire_node("mul", mul(), &[x, two])?[0];
    let z = gpu_model.wire_node("sig", sigmoid(), &[y])?[0];
    gpu_model.select_output_outlets(&[z])?;

    let cpu_model = gpu_model.clone();
    WgpuTransform.transform(&mut gpu_model)?;
    gpu_model = gpu_model.into_optimized()?;
    let gpu_plan = TypedSimplePlan::build(
        gpu_model,
        &RunOptions { skip_order_opt_ram: true, ..RunOptions::default() },
    )?;

    let cpu_plan = SimplePlan::new(cpu_model)?;
    let input =
        Tensor::from_shape(&[2, 4], &(0..8).map(|i| (i as f32) * 0.5 - 1.0).collect::<Vec<_>>())?;
    let gpu_out = Arc::new(gpu_plan).run(tvec!(input.clone().into()))?.remove(0).into_tensor();
    let cpu_out = Arc::new(cpu_plan).run(tvec!(input.into()))?.remove(0).into_tensor();
    close(&gpu_out, &cpu_out)
}

fn run_vs_cpu(mut model: TypedModel, input: Tensor) -> TractResult<()> {
    crate::context::wgpu_context();
    let cpu_model = model.clone();
    WgpuTransform.transform(&mut model)?;
    model = model.into_optimized()?;
    crate::ensure_wgpu_coverage(&model)?;
    let gpu_plan = TypedSimplePlan::build(
        model,
        &RunOptions { skip_order_opt_ram: true, ..RunOptions::default() },
    )?;
    let cpu_plan = SimplePlan::new(cpu_model)?;
    let gpu_out = Arc::new(gpu_plan).run(tvec!(input.clone().into()))?.remove(0).into_tensor();
    let cpu_out = Arc::new(cpu_plan).run(tvec!(input.into()))?.remove(0).into_tensor();
    close(&gpu_out, &cpu_out)
}

#[test]
fn reduce_sum_matches_cpu() -> TractResult<()> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4, 8]))?;
    let y = model.wire_node(
        "sum",
        tract_core::ops::nn::Reduce::new(tvec![2], tract_core::ops::nn::Reducer::Sum),
        &[x],
    )?[0];
    model.select_output_outlets(&[y])?;
    let input = Tensor::from_shape(
        &[2, 4, 8],
        &(0..64).map(|i| (i as f32) * 0.1 - 2.0).collect::<Vec<_>>(),
    )?;
    run_vs_cpu(model, input)
}

#[test]
fn max_pool_2d_nchw_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{MaxPool, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 1, 4, 4]))?;
    let y = model.wire_node(
        "pool",
        MaxPool {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![2, 2],
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: Some(tvec![2, 2]),
                input_channels: 1,
                output_channels: 1,
            },
            with_index_outputs: None,
        },
        &[x],
    )?[0];
    model.select_output_outlets(&[y])?;
    let input = Tensor::from_shape(&[1, 1, 4, 4], &(0..16).map(|i| i as f32).collect::<Vec<_>>())?;
    run_vs_cpu(model, input)
}

#[test]
fn conv2d_nchw_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 1, 4, 4]))?;
    let k = model.add_const(
        "k",
        Tensor::from_shape(&[1, 1, 3, 3], &(0..9).map(|i| i as f32 * 0.1).collect::<Vec<_>>())?,
    )?;
    let b = model.add_const("b", tensor0(0.0f32))?;
    let y = model.wire_node(
        "conv",
        Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![3, 3],
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: None,
                input_channels: 1,
                output_channels: 1,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        },
        &[x, k, b],
    )?[0];
    model.select_output_outlets(&[y])?;
    let input =
        Tensor::from_shape(&[1, 1, 4, 4], &(0..16).map(|i| i as f32 * 0.25).collect::<Vec<_>>())?;
    run_vs_cpu(model, input)
}

#[test]
fn relu_via_max_matches_cpu() -> TractResult<()> {
    use tract_core::ops::math::max;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4]))?;
    let z = model.add_const("zero", tensor2(&[[0.0f32]]))?;
    let y = model.wire_node("relu", max(), &[x, z])?[0];
    model.select_output_outlets(&[y])?;
    let input = Tensor::from_shape(&[2, 4], &(0..8).map(|i| (i as f32) - 3.5).collect::<Vec<_>>())?;
    run_vs_cpu(model, input)
}

#[test]
fn conv_transpose2d_nchw_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 1, 2, 2]))?;
    let k = model.add_const("k", Tensor::from_shape(&[1, 1, 2, 2], &[1.0f32, 0.0, 0.0, 1.0])?)?;
    let b = model.add_const("b", tensor0(0.0f32))?;
    let y = model.wire_node(
        "deconv",
        Deconv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![2, 2],
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: Some(tvec![2, 2]),
                input_channels: 1,
                output_channels: 1,
            },
            kernel_format: KernelFormat::OIHW,
            adjustments: tvec![0, 0],
            group: 1,
        },
        &[x, k, b],
    )?[0];
    model.select_output_outlets(&[y])?;
    let input = Tensor::from_shape(&[1, 1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0])?;
    run_vs_cpu(model, input)
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
fn softmax_matches_cpu() -> TractResult<()> {
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([2, 4]))?;
    let y = model.wire_node(
        "sm",
        tract_core::ops::nn::Softmax::new(
            tvec![1],
            None,
            tract_core::ops::nn::SoftmaxKind::Softmax,
        ),
        &[x],
    )?[0];
    model.select_output_outlets(&[y])?;
    let input = Tensor::from_shape(&[2, 4], &[0.1f32, 0.2, 0.3, 0.4, -1.0, 0.0, 1.0, 2.0])?;
    run_vs_cpu(model, input)
}

fn ramp(shape: &[usize], scale: f32, bias: f32) -> TractResult<Tensor> {
    let n: usize = shape.iter().product();
    Tensor::from_shape(shape, &(0..n).map(|i| (i as f32) * scale + bias).collect::<Vec<_>>())
}

fn matmul_model(
    a_shape: &[usize],
    b_shape: &[usize],
    transpose_a: bool,
    transpose_b: bool,
    transpose_c: bool,
) -> TractResult<TypedModel> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact(a_shape))?;
    let b = model.add_const("b", ramp(b_shape, 0.05, -0.4)?)?;
    let op = PrefixMatMul {
        transpose_a,
        transpose_b,
        transpose_c,
        quantize_output: None,
        operating_dt: None,
    };
    let y = model.wire_node("mm", op, &[a, b])?[0];
    model.select_output_outlets(&[y])?;
    Ok(model)
}

#[test]
fn matmul_2d_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[3, 4], &[4, 5], false, false, false)?;
    run_vs_cpu(model, ramp(&[3, 4], 0.1, -0.5)?)
}

#[test]
fn matmul_transposed_inputs_match_cpu() -> TractResult<()> {
    let model = matmul_model(&[4, 3], &[5, 4], true, true, false)?;
    run_vs_cpu(model, ramp(&[4, 3], 0.1, -0.5)?)
}

#[test]
fn matmul_transposed_output_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[3, 4], &[4, 5], false, false, true)?;
    run_vs_cpu(model, ramp(&[3, 4], 0.1, -0.5)?)
}

/// The prefix broadcasts: one weight matrix against a batch of activations.
#[test]
fn matmul_broadcast_prefix_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[2, 3, 4], &[1, 4, 5], false, false, false)?;
    run_vs_cpu(model, ramp(&[2, 3, 4], 0.05, -0.3)?)
}

#[test]
fn matmul_rank4_prefix_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[2, 3, 4, 5], &[2, 1, 5, 6], false, false, false)?;
    run_vs_cpu(model, ramp(&[2, 3, 4, 5], 0.02, -0.6)?)
}

/// EinSum is what an ONNX pointwise convolution actually arrives as; the
/// transform has to rewrite it to PrefixMatMul before it can reach the GPU.
#[test]
fn einsum_matmul_matches_cpu() -> TractResult<()> {
    use tract_core::ops::einsum::EinSum;
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact([2, 3, 4]))?;
    let b = model.add_const("b", ramp(&[4, 5], 0.05, -0.4)?)?;
    let y =
        model.wire_node("es", EinSum::new("bij,jk->bik".parse()?, f32::datum_type()), &[a, b])?[0];
    model.select_output_outlets(&[y])?;
    run_vs_cpu(model, ramp(&[2, 3, 4], 0.05, -0.3)?)
}

/// Sizes that are not round, so the kernel's bounds checks matter.
#[test]
fn matmul_large_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[40, 33], &[33, 48], false, false, false)?;
    run_vs_cpu(model, ramp(&[40, 33], 0.001, -0.5)?)
}

#[test]
fn matmul_large_transposed_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[33, 40], &[48, 33], true, true, false)?;
    run_vs_cpu(model, ramp(&[33, 40], 0.001, -0.5)?)
}

#[test]
fn matmul_large_batched_matches_cpu() -> TractResult<()> {
    let model = matmul_model(&[2, 35, 33], &[1, 33, 40], false, false, false)?;
    run_vs_cpu(model, ramp(&[2, 35, 33], 0.001, -0.4)?)
}

/// Bias then activation after a matmul: both should land in the GEMM's epilogue.
#[test]
fn matmul_bias_activation_matches_cpu() -> TractResult<()> {
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    let mut model = TypedModel::default();
    let a = model.add_source("a", f32::fact([6, 5]))?;
    let b = model.add_const("b", ramp(&[5, 7], 0.05, -0.4)?)?;
    let bias = model.add_const("bias", ramp(&[1, 7], 0.1, -0.2)?)?;
    let op = PrefixMatMul {
        transpose_a: false,
        transpose_b: false,
        transpose_c: false,
        quantize_output: None,
        operating_dt: None,
    };
    let mm = model.wire_node("mm", op, &[a, b])?[0];
    let biased = model.wire_node("bias_add", tract_core::ops::math::add(), &[mm, bias])?[0];
    let y = model.wire_node("act", sigmoid(), &[biased])?[0];
    model.select_output_outlets(&[y])?;
    run_vs_cpu(model, ramp(&[6, 5], 0.1, -0.5)?)
}

/// Same for a convolution's per-channel bias.
#[test]
fn conv_bias_activation_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    let mut model = TypedModel::default();
    let x = model.add_source("x", f32::fact([1, 2, 5, 5]))?;
    let w = model.add_const("w", ramp(&[3, 2, 3, 3], 0.01, -0.2)?)?;
    let bias = model.add_const("bias", ramp(&[3], 0.3, -0.4)?)?;
    let conv = Conv {
        pool_spec: PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec![3, 3],
            padding: PaddingSpec::Valid,
            dilations: None,
            strides: None,
            input_channels: 2,
            output_channels: 3,
        },
        kernel_fmt: KernelFormat::OIHW,
        group: 1,
        q_params: None,
    };
    let zero = model.add_const("zero", Tensor::zero::<f32>(&[3])?)?;
    let c = model.wire_node("conv", conv, &[x, w, zero])?[0];
    let reshaped = model.wire_node(
        "bias_r",
        AxisOp::Reshape(0, tvec!(3.into()), tvec!(1.into(), 3.into(), 1.into(), 1.into())),
        &[bias],
    )?[0];
    let biased = model.wire_node("bias_add", tract_core::ops::math::add(), &[c, reshaped])?[0];
    let y = model.wire_node("act", sigmoid(), &[biased])?[0];
    model.select_output_outlets(&[y])?;
    run_vs_cpu(model, ramp(&[1, 2, 5, 5], 0.05, -0.5)?)
}

/// Depthwise: one filter per channel, which takes the tiled kernel.
#[test]
fn depthwise_conv2d_matches_cpu() -> TractResult<()> {
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    for (kh, kw, stride, pad) in [
        (3usize, 3usize, 1usize, PaddingSpec::SameUpper),
        (3, 3, 2, PaddingSpec::SameUpper),
        (5, 5, 1, PaddingSpec::SameUpper),
    ] {
        let c = 6;
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, c, 19, 23]))?;
        let w = model.add_const("w", ramp(&[c, 1, kh, kw], 0.01, -0.15)?)?;
        let b = model.add_const("b", Tensor::zero::<f32>(&[c])?)?;
        let y = model.wire_node(
            "dw",
            Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![kh, kw],
                    padding: pad,
                    dilations: None,
                    strides: Some(tvec![stride, stride]),
                    input_channels: c,
                    output_channels: c,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: c,
                q_params: None,
            },
            &[x, w, b],
        )?[0];
        model.select_output_outlets(&[y])?;
        run_vs_cpu(model, ramp(&[1, c, 19, 23], 0.003, -0.4)?)?;
    }
    Ok(())
}

/// Does one compute pass run independent dispatches concurrently? A chain of
/// dependent kernels has to serialize; the same count over disjoint buffers
/// does not. If both take the same time, nothing overlaps.
///
///     cargo test --release -p tract-wgpu -- --ignored --nocapture concurrency
#[test]
#[ignore]
fn dispatch_concurrency() -> TractResult<()> {
    with_wgpu_queue(|q| {
        for count in [1, 16, 64, 256] {
            dispatch_concurrency_at(q, count, 1 << 10)?;
        }
        for shift in [8, 12, 16] {
            dispatch_concurrency_at(q, 64, 1 << shift)?;
        }
        Ok(())
    })
}

#[cfg(test)]
fn dispatch_concurrency_at(q: &crate::WgpuQueue, count: usize, len: usize) -> TractResult<()> {
    {
        use std::time::Instant;
        let host = Tensor::from_shape(&[len], &vec![0.25f32; len])?;
        let inputs: Vec<DeviceTensor> =
            (0..count).map(|_| host.clone().into_device()).collect::<TractResult<_>>()?;
        let outputs: Vec<DeviceTensor> = (0..count)
            .map(|_| DeviceTensor::uninitialized_dt(DatumType::F32, &[len]))
            .collect::<TractResult<_>>()?;
        let op = sigmoid();

        let independent = || -> TractResult<()> {
            for i in 0..count {
                wgpu_element_wise_dispatch(&*op.0, &inputs[i], &outputs[i])?;
            }
            q.flush()
        };
        let chained = || -> TractResult<()> {
            for i in 0..count {
                let src = if i == 0 { &inputs[0] } else { &outputs[i - 1] };
                wgpu_element_wise_dispatch(&*op.0, src, &outputs[i])?;
            }
            q.flush()
        };

        for _ in 0..3 {
            independent()?;
            chained()?;
        }
        let record_only = || -> TractResult<f64> {
            let t = Instant::now();
            for i in 0..count {
                wgpu_element_wise_dispatch(&*op.0, &inputs[i], &outputs[i])?;
            }
            let recorded = t.elapsed().as_secs_f64() * 1e3;
            q.flush()?;
            Ok(recorded)
        };
        let mut best = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..10 {
            let t = Instant::now();
            independent()?;
            best.0 = best.0.min(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            chained()?;
            best.1 = best.1.min(t.elapsed().as_secs_f64() * 1e3);
            best.2 = best.2.min(record_only()?);
        }
        eprintln!(
            "{count} dispatches of {len:>6} elements ({:>4} workgroups each): independent {:.3}ms, chain {:.3}ms, cpu-side recording {:.3}ms",
            len.div_ceil(64),
            best.0,
            best.1,
            best.2
        );
        Ok(())
    }
}

/// What a frame pays per node before any kernel runs: allocating the output
/// tensor, and recording one dispatch against it.
///
///     cargo test --release -p tract-wgpu -- --ignored --nocapture per_node_cost
#[test]
#[ignore]
fn per_node_cost() -> TractResult<()> {
    use std::time::Instant;
    with_wgpu_queue(|q| {
        let shape = [1usize, 64, 144, 256];
        let op = sigmoid();
        let input = Tensor::zero::<f32>(&shape)?.into_device()?;
        // Warm the pool so the allocation path is the steady-state one.
        for _ in 0..64 {
            let _ = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
        }
        let n = 2000;
        let t = Instant::now();
        for _ in 0..n {
            let _out = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
        }
        let alloc = t.elapsed().as_secs_f64() * 1e6 / n as f64;

        let out = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
        let t = Instant::now();
        for _ in 0..n {
            wgpu_element_wise_dispatch(&*op.0, &input, &out)?;
        }
        let record = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        q.flush()?;

        let t = Instant::now();
        for _ in 0..n {
            let out = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
            wgpu_element_wise_dispatch(&*op.0, &input, &out)?;
        }
        let both = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        q.flush()?;
        eprintln!("per node: allocate {alloc:.2}us, record {record:.2}us, together {both:.2}us");
        Ok(())
    })
}

/// What building a bind group costs, against finding one already built.
///
///     cargo test --release -p tract-wgpu -- --ignored --nocapture bind_group_cost
#[test]
#[ignore]
fn bind_group_cost() -> TractResult<()> {
    use crate::kernels::shaders::LayoutKind;
    use std::time::Instant;
    with_wgpu_queue(|q| {
        let ctx = q.context();
        let n = 2000;
        let pairs: Vec<(DeviceTensor, DeviceTensor)> = (0..n)
            .map(|_| {
                Ok((
                    DeviceTensor::uninitialized_dt(DatumType::F32, &[1024])?,
                    DeviceTensor::uninitialized_dt(DatumType::F32, &[1024])?,
                ))
            })
            .collect::<TractResult<_>>()?;

        let t = Instant::now();
        for (a, b) in &pairs {
            let _ = ctx.bind_group(
                LayoutKind::Unary,
                &[crate::utils::get_wgpu_buffer(a), crate::utils::get_wgpu_buffer(b)],
                q.uniform(),
            )?;
        }
        let miss = t.elapsed().as_secs_f64() * 1e6 / n as f64;

        let (a, b) = &pairs[0];
        let t = Instant::now();
        for _ in 0..n {
            let _ = ctx.bind_group(
                LayoutKind::Unary,
                &[crate::utils::get_wgpu_buffer(a), crate::utils::get_wgpu_buffer(b)],
                q.uniform(),
            )?;
        }
        let hit = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        eprintln!("bind group: {miss:.2}us to build, {hit:.2}us to find");
        Ok(())
    })
}

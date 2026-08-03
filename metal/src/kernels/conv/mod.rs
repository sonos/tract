pub mod mlx_conv;

use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

pub fn kernel_name(hw_rank: usize, dt: DatumType) -> TractResult<String> {
    let dt_name = if dt == DatumType::F16 { "f16" } else { "f32" };
    Ok(format!("conv{hw_rank}d_{dt_name}_generic"))
}

pub fn metal_conv_direct(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    output: &DeviceTensor,
) -> TractResult<()> {
    metal_conv_dispatch_inner(stream, op, input, weights, bias, output)
}

pub fn metal_conv_dispatch(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    output: &DeviceTensor,
) -> TractResult<()> {
    // The MLX kernel takes bias separately (it is added by a following op), so
    // it only handles the bias-free dispatch here.
    if bias.is_none() {
        if mlx_conv::mlx_depthwise_dispatchable(op, input, weights) {
            return mlx_conv::dispatch_mlx_depthwise_conv_2d(stream, op, input, weights, output);
        }
        if mlx_conv::mlx_conv_dispatchable(op, input, weights) {
            return mlx_conv::dispatch_mlx_conv_2d(stream, op, input, weights, output);
        }
    }
    metal_conv_dispatch_inner(stream, op, input, weights, bias, output)
}

fn metal_conv_dispatch_inner(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(weights);
    if let Some(b) = bias {
        stream.retain_tensor(b);
    }
    stream.retain_tensor(output);

    let input_shape = op.pool_spec.data_format.shape(input.shape())?;
    let hw_rank = input_shape.hw_rank();
    let func_name = kernel_name(hw_rank, input.datum_type())?;
    let pipeline = stream.load_pipeline(LibraryName::ConvOps, &func_name)?;

    let co_per_group = op.pool_spec.output_channels / op.group;
    let ci_per_group = op.pool_spec.input_channels / op.group;

    // in_shape: [N, C, spatial...]
    let in_n = *input_shape.n().unwrap_or(&1);
    let in_c = *input_shape.c();
    let mut in_shape_buf: TVec<i32> = tvec![in_n as i32, in_c as i32];
    in_shape_buf.extend(input_shape.hw_dims().iter().map(|&d| d as i32));

    let mut in_strides_buf: TVec<i32> =
        tvec![*input_shape.n_stride().unwrap_or(&0) as i32, *input_shape.c_stride() as i32];
    in_strides_buf.extend(input_shape.hw_strides().iter().map(|&s| s as i32));

    // ker_params: [groups, co_per_group, ci_per_group, ker_spatial...]
    let mut ker_params: TVec<i32> =
        tvec![op.group as i32, co_per_group as i32, ci_per_group as i32];
    ker_params.extend(weights.shape()[2..].iter().map(|&d| d as i32));

    // ker_strides: [g_stride, o_stride, i_stride, spatial...]
    let group_stride = weights.strides()[0] as usize * co_per_group;
    let mut ker_strides: TVec<i32> = tvec![group_stride as i32];
    ker_strides.extend(weights.strides().iter().map(|&s| s as i32));

    // padding
    let padding = op.pool_spec.computed_padding(input_shape.hw_dims());
    let pad_buf: TVec<i32> = padding.iter().map(|p| p.pad_before as i32).collect();

    let strides = op.pool_spec.strides();
    let strides_buf: TVec<i32> = strides.iter().map(|&s| s as i32).collect();

    let dilations = op.pool_spec.dilations();
    let dilations_buf: TVec<i32> = dilations.iter().map(|&d| d as i32).collect();

    let output_shape = op.pool_spec.data_format.shape(output.shape())?;
    let out_n = *output_shape.n().unwrap_or(&1);
    let out_c = *output_shape.c();
    let mut out_shape_buf: TVec<i32> = tvec![out_n as i32, out_c as i32];
    out_shape_buf.extend(output_shape.hw_dims().iter().map(|&d| d as i32));

    let mut out_strides_buf: TVec<i32> =
        tvec![*output_shape.n_stride().unwrap_or(&0) as i32, *output_shape.c_stride() as i32];
    out_strides_buf.extend(output_shape.hw_strides().iter().map(|&s| s as i32));

    // bias_stride: -1 means no bias, 0 means scalar broadcast, 1 means per-channel
    let bias_stride: i32 = if let Some(b) = bias { if b.rank() == 0 { 0 } else { 1 } } else { -1 };

    let spatial_out: usize = output_shape.hw_dims().iter().product();
    let threads_per_group = 32usize;

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_slice(1, &in_shape_buf);
        encoder.set_slice(2, &in_strides_buf);
        encoder.set_metal_tensor(3, weights, metal::MTLResourceUsage::Read);
        encoder.set_slice(4, &ker_params);
        encoder.set_slice(5, &ker_strides);
        if let Some(b) = bias {
            encoder.set_metal_tensor(6, b, metal::MTLResourceUsage::Read);
        } else {
            // Empty buffer — kernel checks bias_stride < 0
            encoder.set_bytes(6, 0, std::ptr::null());
        }
        encoder.set_slice(7, &[bias_stride]);
        encoder.set_slice(8, &pad_buf);
        encoder.set_slice(9, &strides_buf);
        encoder.set_slice(10, &dilations_buf);
        encoder.set_metal_tensor(11, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(12, &out_shape_buf);
        encoder.set_slice(13, &out_strides_buf);

        let grid_size = MTLSize {
            width: spatial_out.div_ceil(threads_per_group) as _,
            height: out_c as _,
            depth: out_n as _,
        };
        let group_size = MTLSize { width: threads_per_group as _, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

#[cfg(test)]
mod mlx_conv_tests {
    use crate::LibraryName;
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::internal::*;

    use super::mlx_conv::dispatch_mlx_conv_2d;
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    use tract_gpu::tensor::{DeviceTensor, IntoDevice};

    #[test]
    fn mlx_conv_library_compiles() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            stream.load_library(LibraryName::MlxConv)?;
            Ok(())
        })
    }

    // Each kernel gets the weight layout it expects (direct: OIHW, MLX: OHWI)
    // for the same convolution, so the two are comparable.
    fn ohwi_to_oihw(w: &Tensor) -> TractResult<Tensor> {
        w.clone().move_axis(3, 1) // O,H,W,I -> O,I,H,W
    }

    fn ramp(dt: DatumType, shape: &[usize], seed: usize) -> TractResult<Tensor> {
        let len: usize = shape.iter().product();
        let v: Vec<f32> =
            (0..len).map(|i| (((i * 13 + seed * 7) % 23) as f32 - 11.0) / 32.0).collect();
        Ok(Tensor::from_shape(shape, &v)?.cast_to_dt(dt)?.into_owned())
    }

    #[allow(clippy::too_many_arguments)]
    fn check_conv(
        dt: DatumType,
        n: usize,
        ih: usize,
        iw: usize,
        c: usize,
        o: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        dil: usize,
        padding: PaddingSpec,
    ) -> TractResult<()> {
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec![kh, kw],
            padding.clone(),
            Some(tvec![dil, dil]),
            Some(tvec![stride, stride]),
            c,
            o,
        );
        let op = Conv { pool_spec, kernel_fmt: KernelFormat::OHWI, group: 1, q_params: None };
        let input = ramp(dt, &[n, ih, iw, c], 1)?;
        let weights = ramp(dt, &[o, kh, kw, c], 2)?;
        let bias = Tensor::zero_dt(dt, &[o])?;
        let expected = op
            .eval(tvec![
                input.clone().into_tvalue(),
                weights.clone().into_tvalue(),
                bias.into_tvalue()
            ])?
            .remove(0)
            .into_tensor();
        let got = with_borrowed_metal_stream(|stream| {
            let i = input.clone().into_device()?;
            let w = weights.clone().into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(dt, expected.shape())? };
            dispatch_mlx_conv_2d(stream, &op, &i, &w, &out)?;
            stream.wait_until_completed()?;
            Ok(out.to_host()?.into_tensor())
        })?;
        expected.close_enough(&got, Approximation::Approximate).with_context(|| {
            format!("dt={dt:?} n={n} {ih}x{iw}x{c} -> {o} k={kh}x{kw} s={stride} d={dil}")
        })
    }

    #[test]
    fn mlx_conv_1x1() -> TractResult<()> {
        check_conv(DatumType::F32, 1, 8, 8, 16, 32, 1, 1, 1, 1, PaddingSpec::Valid)
    }

    #[test]
    fn mlx_conv_3x3_valid() -> TractResult<()> {
        check_conv(DatumType::F32, 1, 10, 10, 16, 16, 3, 3, 1, 1, PaddingSpec::Valid)
    }

    #[test]
    fn mlx_conv_3x3_same() -> TractResult<()> {
        check_conv(DatumType::F32, 2, 9, 7, 32, 16, 3, 3, 1, 1, PaddingSpec::SameUpper)
    }

    #[test]
    fn mlx_conv_strided() -> TractResult<()> {
        check_conv(DatumType::F32, 1, 16, 16, 16, 32, 3, 3, 2, 1, PaddingSpec::SameUpper)
    }

    #[test]
    fn mlx_conv_dilated() -> TractResult<()> {
        check_conv(DatumType::F32, 1, 16, 16, 16, 16, 3, 3, 1, 2, PaddingSpec::Valid)
    }

    #[test]
    fn mlx_conv_f16() -> TractResult<()> {
        check_conv(DatumType::F16, 1, 12, 12, 32, 64, 3, 3, 1, 1, PaddingSpec::SameUpper)
    }

    #[test]
    fn mlx_conv_unaligned_channels() -> TractResult<()> {
        check_conv(DatumType::F32, 1, 8, 8, 5, 7, 3, 3, 1, 1, PaddingSpec::Valid)
    }

    // MLX implicit-GEMM conv against the direct kernel, on shapes a vision
    // model actually runs.
    //   cargo test -p tract-metal bench_conv -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_conv() -> TractResult<()> {
        use std::time::Instant;
        println!("\n  shape (N,H,W,C -> O, kHxkW, s)          direct ms   mlx ms    gain");
        for &(n, ih, iw, c, o, k, stride) in &[
            (1usize, 112usize, 112usize, 32usize, 64usize, 3usize, 1usize),
            (1, 56, 56, 64, 128, 3, 1),
            (1, 28, 28, 128, 256, 3, 1),
            (1, 56, 56, 64, 64, 1, 1),
            (1, 224, 224, 3, 32, 3, 2),
            (8, 28, 28, 128, 128, 3, 1),
        ] {
            for dt in [DatumType::F32, DatumType::F16] {
                let pool_spec = PoolSpec::new(
                    DataFormat::NHWC,
                    tvec![k, k],
                    PaddingSpec::SameUpper,
                    Some(tvec![1, 1]),
                    Some(tvec![stride, stride]),
                    c,
                    o,
                );
                let op = Conv {
                    pool_spec: pool_spec.clone(),
                    kernel_fmt: KernelFormat::OHWI,
                    group: 1,
                    q_params: None,
                };
                let op_dir =
                    Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };
                let input = ramp(dt, &[n, ih, iw, c], 1)?;
                let weights = ramp(dt, &[o, k, k, c], 2)?;
                let weights_oihw = ohwi_to_oihw(&weights)?;
                let oh = (ih + stride - 1) / stride;
                let ow = (iw + stride - 1) / stride;
                let (direct, mlx) = with_borrowed_metal_stream(|stream| {
                    let i = input.clone().into_device()?;
                    let w = weights.clone().into_device()?;
                    let w_dir = weights_oihw.clone().into_device()?;
                    let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[n, oh, ow, o])? };
                    let time = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
                        for _ in 0..3 {
                            f()?;
                        }
                        stream.wait_until_completed()?;
                        let mut best = f64::MAX;
                        for _ in 0..5 {
                            let t = Instant::now();
                            for _ in 0..10 {
                                f()?;
                            }
                            stream.wait_until_completed()?;
                            best = best.min(t.elapsed().as_secs_f64() / 10.0);
                        }
                        Ok(best)
                    };
                    let d = time(&|| {
                        super::metal_conv_direct(stream, &op_dir, &i, &w_dir, None, &out)
                    })?;
                    let m =
                        time(&|| super::mlx_conv::dispatch_mlx_conv_2d(stream, &op, &i, &w, &out))?;
                    Ok((d, m))
                })?;
                println!(
                    "  {dt:?} {n}x{ih}x{iw}x{c} -> {o}, {k}x{k}, s{stride}   {:9.4} {:9.4}  {:6.2}x",
                    direct * 1e3,
                    mlx * 1e3,
                    direct / mlx
                );
            }
        }
        Ok(())
    }

    // Sanity: both kernels must agree with the CPU op, so the bench compares
    // two correct implementations.
    #[test]
    fn direct_and_mlx_agree_with_cpu() -> TractResult<()> {
        let dt = DatumType::F32;
        let (n, ih, iw, c, o, k) = (1usize, 16usize, 16usize, 32usize, 32usize, 3usize);
        let spec = |fmt: KernelFormat| {
            let pool_spec = PoolSpec::new(
                DataFormat::NHWC,
                tvec![k, k],
                PaddingSpec::SameUpper,
                Some(tvec![1, 1]),
                Some(tvec![1, 1]),
                c,
                o,
            );
            Conv { pool_spec, kernel_fmt: fmt, group: 1, q_params: None }
        };
        let op_mlx = spec(KernelFormat::OHWI);
        let op_dir = spec(KernelFormat::OIHW);
        let input = ramp(dt, &[n, ih, iw, c], 1)?;
        let w_ohwi = ramp(dt, &[o, k, k, c], 2)?;
        let w_oihw = ohwi_to_oihw(&w_ohwi)?;
        let bias = Tensor::zero_dt(dt, &[o])?;
        let expected = op_mlx
            .eval(tvec![
                input.clone().into_tvalue(),
                w_ohwi.clone().into_tvalue(),
                bias.clone().into_tvalue()
            ])?
            .remove(0)
            .into_tensor();
        // the transposed weights must describe the same convolution
        let via_oihw = op_dir
            .eval(tvec![
                input.clone().into_tvalue(),
                w_oihw.clone().into_tvalue(),
                bias.into_tvalue()
            ])?
            .remove(0)
            .into_tensor();
        expected.close_enough(&via_oihw, Approximation::Approximate)?;

        with_borrowed_metal_stream(|stream| {
            let i = input.clone().into_device()?;
            let wd = w_oihw.clone().into_device()?;
            let wm = w_ohwi.clone().into_device()?;
            let od = unsafe { DeviceTensor::uninitialized_dt(dt, expected.shape())? };
            super::metal_conv_direct(stream, &op_dir, &i, &wd, None, &od)?;
            stream.wait_until_completed()?;
            expected
                .close_enough(&od.to_host()?.into_tensor(), Approximation::Approximate)
                .context("direct kernel")?;
            let om = unsafe { DeviceTensor::uninitialized_dt(dt, expected.shape())? };
            super::mlx_conv::dispatch_mlx_conv_2d(stream, &op_mlx, &i, &wm, &om)?;
            stream.wait_until_completed()?;
            expected
                .close_enough(&om.to_host()?.into_tensor(), Approximation::Approximate)
                .context("mlx kernel")?;
            Ok(())
        })
    }

    // End to end through the metal transform: an NHWC conv must reach the MLX
    // kernel and match the CPU result.
    #[test]
    fn conv_routes_through_metal_transform() -> TractResult<()> {
        use crate::MetalTransform;
        use tract_core::transform::ModelTransform;
        let dt = DatumType::F32;
        let (n, ih, iw, c, o, k) = (1usize, 12usize, 12usize, 16usize, 32usize, 3usize);
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec![k, k],
            PaddingSpec::SameUpper,
            Some(tvec![1, 1]),
            Some(tvec![1, 1]),
            c,
            o,
        );
        // an exported model arrives in OIHW; the metal rule must flip it
        let op = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };
        let input = ramp(dt, &[n, ih, iw, c], 1)?;
        let w_oihw = ohwi_to_oihw(&ramp(dt, &[o, k, k, c], 2)?)?;
        let bias = Tensor::zero_dt(dt, &[o])?;
        let mut model = TypedModel::default();
        let i = model.add_source("i", dt.fact(&[n, ih, iw, c]))?;
        let w = model.add_const("w", w_oihw.clone())?;
        let b = model.add_const("b", bias.clone())?;
        let out = model.wire_node("conv", op.clone(), &[i, w, b])?;
        model.select_output_outlets(&out)?;
        let cpu = model.clone().into_runnable()?.run(tvec![input.clone().into_tvalue()])?;

        let metal = MetalTransform::default().transform_into(model)?;
        let ohwi = metal.nodes().iter().any(|n| {
            n.op_as::<crate::ops::conv::MetalConv>()
                .map(|c| c.op.kernel_fmt == KernelFormat::OHWI)
                .unwrap_or(false)
        });
        assert!(ohwi, "conv should have been moved to OHWI for the MLX kernel");
        let got = metal.into_runnable()?.run(tvec![input.into_tvalue()])?;
        cpu[0]
            .clone()
            .into_tensor()
            .close_enough(&got[0].clone().into_tensor(), Approximation::Approximate)?;
        Ok(())
    }

    // How long each conv pipeline specialization takes to build.
    //   cargo test -p tract-metal bench_conv_pipeline_build -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_conv_pipeline_build() -> TractResult<()> {
        use crate::{ConstantValues, Value};
        use std::time::Instant;
        with_borrowed_metal_stream(|stream| {
            let t = Instant::now();
            stream.load_library(LibraryName::MlxConv)?;
            println!("  library compile            {:8.1} ms", t.elapsed().as_secs_f64() * 1e3);
            for (bm, bn) in [(32, 32), (64, 32), (32, 64), (64, 64)] {
                for align in [true, false] {
                    for tname in ["float32", "float16"] {
                        let name = format!(
                            "implicit_gemm_conv_2d_general_{tname}_bm{bm}_bn{bn}_bk16_wm2_wn2"
                        );
                        let consts = Some(ConstantValues::new(vec![(200, Value::Bool(align))]));
                        let t = Instant::now();
                        stream.load_pipeline_with_constants(LibraryName::MlxConv, &name, consts)?;
                        println!(
                            "  {tname} bm{bm} bn{bn} alC={align:<5}  {:8.1} ms",
                            t.elapsed().as_secs_f64() * 1e3
                        );
                    }
                }
            }
            Ok(())
        })
    }

    // Inception v3 conv shapes, to find which one the ported kernel handles badly.
    //   cargo test -p tract-metal bench_inception_shapes -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_inception_shapes() -> TractResult<()> {
        use std::time::Instant;
        let dt = DatumType::F32;
        // (ih, iw, c, o, kh, kw, stride, valid)
        for &(ih, iw, c, o, kh, kw, st, valid) in &[
            (299usize, 299usize, 3usize, 32usize, 3usize, 3usize, 2usize, true),
            (149, 149, 32, 32, 3, 3, 1, true),
            (147, 147, 32, 64, 3, 3, 1, false),
            (73, 73, 80, 192, 3, 3, 1, true),
            (35, 35, 48, 64, 5, 5, 1, false),
            (35, 35, 64, 96, 3, 3, 1, false),
            (17, 17, 128, 128, 1, 7, 1, false),
            (17, 17, 128, 192, 7, 1, 1, false),
            (8, 8, 384, 384, 1, 3, 1, false),
        ] {
            let pool_spec = PoolSpec::new(
                DataFormat::NHWC,
                tvec![kh, kw],
                if valid { PaddingSpec::Valid } else { PaddingSpec::SameUpper },
                Some(tvec![1, 1]),
                Some(tvec![st, st]),
                c,
                o,
            );
            let op = Conv {
                pool_spec: pool_spec.clone(),
                kernel_fmt: KernelFormat::OHWI,
                group: 1,
                q_params: None,
            };
            let op_dir =
                Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };
            let input = ramp(dt, &[1, ih, iw, c], 1)?;
            let w = ramp(dt, &[o, kh, kw, c], 2)?;
            let w_dir = ohwi_to_oihw(&w)?;
            let oh = if valid { (ih - kh) / st + 1 } else { (ih + st - 1) / st };
            let ow = if valid { (iw - kw) / st + 1 } else { (iw + st - 1) / st };
            let (d, m) = with_borrowed_metal_stream(|stream| {
                let i = input.clone().into_device()?;
                let wm = w.clone().into_device()?;
                let wd = w_dir.clone().into_device()?;
                let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[1, oh, ow, o])? };
                let time = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
                    f()?;
                    stream.wait_until_completed()?;
                    let t = Instant::now();
                    for _ in 0..5 {
                        f()?;
                    }
                    stream.wait_until_completed()?;
                    Ok(t.elapsed().as_secs_f64() / 5.0)
                };
                let d = time(&|| super::metal_conv_direct(stream, &op_dir, &i, &wd, None, &out))?;
                let m =
                    time(&|| super::mlx_conv::dispatch_mlx_conv_2d(stream, &op, &i, &wm, &out))?;
                Ok((d, m))
            })?;
            println!(
                "  {ih}x{iw}x{c} -> {o}, {kh}x{kw}, s{st}{}  direct {:9.3} ms   mlx {:9.3} ms  {:7.2}x",
                if valid { " valid" } else { " same " },
                d * 1e3,
                m * 1e3,
                d / m
            );
        }
        Ok(())
    }

    // The kernel arrives in whatever layout the exporter used — TF gives HWIO,
    // ONNX gives OIHW — and the ported kernel indexes OHWI. Each must reach it
    // correctly; getting this wrong reads garbage loop bounds, not a crash.
    #[test]
    fn every_kernel_format_reaches_the_mlx_kernel() -> TractResult<()> {
        use crate::MetalTransform;
        use tract_core::transform::ModelTransform;
        let dt = DatumType::F32;
        let (n, ih, iw, c, o, kh, kw) =
            (1usize, 12usize, 12usize, 16usize, 32usize, 3usize, 3usize);
        let w_ohwi = ramp(dt, &[o, kh, kw, c], 2)?;
        for fmt in [KernelFormat::OHWI, KernelFormat::OIHW, KernelFormat::HWIO] {
            let weights = match fmt {
                KernelFormat::OHWI => w_ohwi.clone(),
                // [O,H,W,I] -> [O,I,H,W]
                KernelFormat::OIHW => w_ohwi.clone().move_axis(3, 1)?,
                // [O,H,W,I] -> [H,W,I,O]
                KernelFormat::HWIO => w_ohwi.clone().move_axis(0, 3)?,
            };
            let pool_spec = PoolSpec::new(
                DataFormat::NHWC,
                tvec![kh, kw],
                PaddingSpec::SameUpper,
                Some(tvec![1, 1]),
                Some(tvec![1, 1]),
                c,
                o,
            );
            let op = Conv { pool_spec, kernel_fmt: fmt, group: 1, q_params: None };
            let input = ramp(dt, &[n, ih, iw, c], 1)?;
            let bias = Tensor::zero_dt(dt, &[o])?;
            let mut model = TypedModel::default();
            let i = model.add_source("i", dt.fact(&[n, ih, iw, c]))?;
            let w = model.add_const("w", weights)?;
            let b = model.add_const("b", bias)?;
            let out = model.wire_node("conv", op, &[i, w, b])?;
            model.select_output_outlets(&out)?;
            let cpu = model.clone().into_runnable()?.run(tvec![input.clone().into_tvalue()])?;
            let metal = MetalTransform::default().transform_into(model)?;
            let got = metal.into_runnable()?.run(tvec![input.clone().into_tvalue()])?;
            cpu[0]
                .clone()
                .into_tensor()
                .close_enough(&got[0].clone().into_tensor(), Approximation::Approximate)
                .with_context(|| format!("kernel format {fmt:?}"))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn check_depthwise(
        dt: DatumType,
        n: usize,
        ih: usize,
        iw: usize,
        c: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: PaddingSpec,
    ) -> TractResult<()> {
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec![kh, kw],
            padding,
            Some(tvec![1, 1]),
            Some(tvec![stride, stride]),
            c,
            c,
        );
        let op = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: c, q_params: None };
        let input = ramp(dt, &[n, ih, iw, c], 1)?;
        let weights = ramp(dt, &[c, 1, kh, kw], 2)?;
        let bias = Tensor::zero_dt(dt, &[c])?;
        let expected = op
            .eval(tvec![
                input.clone().into_tvalue(),
                weights.clone().into_tvalue(),
                bias.into_tvalue()
            ])?
            .remove(0)
            .into_tensor();
        let got = with_borrowed_metal_stream(|stream| {
            let i = input.clone().into_device()?;
            let w = weights.clone().into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(dt, expected.shape())? };
            super::mlx_conv::dispatch_mlx_depthwise_conv_2d(stream, &op, &i, &w, &out)?;
            stream.wait_until_completed()?;
            Ok(out.to_host()?.into_tensor())
        })?;
        expected.close_enough(&got, Approximation::Approximate).with_context(|| {
            format!("depthwise dt={dt:?} {n}x{ih}x{iw}x{c} k={kh}x{kw} s={stride}")
        })
    }

    #[test]
    fn depthwise_3x3_same() -> TractResult<()> {
        check_depthwise(DatumType::F32, 1, 14, 14, 32, 3, 3, 1, PaddingSpec::SameUpper)
    }

    #[test]
    fn depthwise_3x3_stride2() -> TractResult<()> {
        check_depthwise(DatumType::F32, 1, 16, 16, 64, 3, 3, 2, PaddingSpec::SameUpper)
    }

    #[test]
    fn depthwise_5x5_valid() -> TractResult<()> {
        check_depthwise(DatumType::F32, 2, 12, 12, 16, 5, 5, 1, PaddingSpec::Valid)
    }

    #[test]
    fn depthwise_f16() -> TractResult<()> {
        check_depthwise(DatumType::F16, 1, 14, 14, 144, 3, 3, 1, PaddingSpec::SameUpper)
    }

    #[test]
    fn depthwise_non_square() -> TractResult<()> {
        check_depthwise(DatumType::F32, 1, 15, 11, 48, 3, 3, 1, PaddingSpec::SameUpper)
    }

    // A depthwise conv coming in as OIHW (the shared layout) must reach the
    // depthwise kernel and match CPU.
    #[test]
    fn depthwise_routes_through_metal_transform() -> TractResult<()> {
        use crate::MetalTransform;
        use tract_core::transform::ModelTransform;
        let dt = DatumType::F32;
        let (n, ih, iw, c, k) = (1usize, 14usize, 14usize, 32usize, 3usize);
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec![k, k],
            PaddingSpec::SameUpper,
            Some(tvec![1, 1]),
            Some(tvec![1, 1]),
            c,
            c,
        );
        let op = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: c, q_params: None };
        let input = ramp(dt, &[n, ih, iw, c], 1)?;
        let weights = ramp(dt, &[c, 1, k, k], 2)?;
        let bias = Tensor::zero_dt(dt, &[c])?;
        let mut model = TypedModel::default();
        let i = model.add_source("i", dt.fact(&[n, ih, iw, c]))?;
        let w = model.add_const("w", weights)?;
        let b = model.add_const("b", bias)?;
        let out = model.wire_node("conv", op, &[i, w, b])?;
        model.select_output_outlets(&out)?;
        let cpu = model.clone().into_runnable()?.run(tvec![input.clone().into_tvalue()])?;
        let metal = MetalTransform::default().transform_into(model)?;
        let got = metal.into_runnable()?.run(tvec![input.into_tvalue()])?;
        cpu[0]
            .clone()
            .into_tensor()
            .close_enough(&got[0].clone().into_tensor(), Approximation::Approximate)?;
        Ok(())
    }

    // MobileNet-class depthwise shapes against the direct kernel.
    //   cargo test -p tract-metal bench_depthwise -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_depthwise() -> TractResult<()> {
        use std::time::Instant;
        println!("\n  shape (N,H,W,C, kxk, s)        direct ms     mlx ms    gain");
        for &(ih, iw, c, k, st) in &[
            (112usize, 112usize, 32usize, 3usize, 1usize),
            (112, 112, 144, 3, 2),
            (56, 56, 192, 3, 1),
            (28, 28, 384, 3, 1),
            (14, 14, 576, 3, 1),
            (7, 7, 960, 3, 1),
        ] {
            for dt in [DatumType::F32, DatumType::F16] {
                let pool_spec = PoolSpec::new(
                    DataFormat::NHWC,
                    tvec![k, k],
                    PaddingSpec::SameUpper,
                    Some(tvec![1, 1]),
                    Some(tvec![st, st]),
                    c,
                    c,
                );
                let op =
                    Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: c, q_params: None };
                let input = ramp(dt, &[1, ih, iw, c], 1)?;
                let weights = ramp(dt, &[c, 1, k, k], 2)?;
                let (oh, ow) = ((ih + st - 1) / st, (iw + st - 1) / st);
                let (d, m) = with_borrowed_metal_stream(|stream| {
                    let i = input.clone().into_device()?;
                    let w = weights.clone().into_device()?;
                    let out = unsafe { DeviceTensor::uninitialized_dt(dt, &[1, oh, ow, c])? };
                    let time = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
                        f()?;
                        stream.wait_until_completed()?;
                        let t = Instant::now();
                        for _ in 0..10 {
                            f()?;
                        }
                        stream.wait_until_completed()?;
                        Ok(t.elapsed().as_secs_f64() / 10.0)
                    };
                    let d = time(&|| super::metal_conv_direct(stream, &op, &i, &w, None, &out))?;
                    let m = time(&|| {
                        super::mlx_conv::dispatch_mlx_depthwise_conv_2d(stream, &op, &i, &w, &out)
                    })?;
                    Ok((d, m))
                })?;
                println!(
                    "  {dt:?} {ih}x{iw}x{c}, {k}x{k}, s{st}    {:9.4} {:9.4}  {:6.2}x",
                    d * 1e3,
                    m * 1e3,
                    d / m
                );
            }
        }
        Ok(())
    }
}

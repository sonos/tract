//! 2D max and sum pooling on channels-last tensors (see pool.metal).
//!
//! Pooling had no Metal kernel, so a pooled model bounced back to the host at
//! every pool: on Inception v3 that was 14 device syncs and a fifth of the
//! runtime.

use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::cnn::PoolSpec;
use tract_gpu::tensor::DeviceTensor;

/// Mirror of `PoolParams` in pool.metal — keep field order in sync.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PoolParams {
    n: i32,
    ih: i32,
    iw: i32,
    c: i32,
    oh: i32,
    ow: i32,
    kh: i32,
    kw: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
    dil_h: i32,
    dil_w: i32,
    count_include_pad: i32,
    normalize: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Max,
    /// `normalize` turns the sum into an average.
    Sum {
        count_include_pad: bool,
        normalize: bool,
    },
}

/// Whether the kernel covers this pooling: rank-4 f16/f32 with static geometry
/// and no index output. Max pooling has a channels-first variant too; sum
/// pooling stays channels-last.
pub fn metal_pool_supported(pool_spec: &PoolSpec, kind: PoolKind, fact: &TypedFact) -> bool {
    let layout_ok = pool_spec.data_format.c_is_last()
        || (kind == PoolKind::Max && pool_spec.data_format.has_n());
    matches!(fact.datum_type, DatumType::F16 | DatumType::F32)
        && layout_ok
        && fact.rank() == 4
        && pool_spec.kernel_shape.len() == 2
        && fact.shape.as_concrete().is_some()
}

pub fn dispatch_metal_pool(
    stream: &MetalStream,
    pool_spec: &PoolSpec,
    kind: PoolKind,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let dt = input.datum_type();
    let tname = match dt {
        DatumType::F32 => "f32",
        DatumType::F16 => "f16",
        _ => bail!("Metal pool: F32/F16 only, got {dt:?}"),
    };
    let in_shape = pool_spec.data_format.shape(input.shape())?;
    let out_shape = pool_spec.data_format.shape(output.shape())?;
    ensure!(in_shape.hw_rank() == 2, "Metal pool is 2D only");

    let strides = pool_spec.strides();
    let dilations = pool_spec.dilations();
    let padding = pool_spec.computed_padding(in_shape.hw_dims());
    let (count_include_pad, normalize) = match kind {
        PoolKind::Max => (false, false),
        PoolKind::Sum { count_include_pad, normalize } => (count_include_pad, normalize),
    };
    let params = PoolParams {
        n: *in_shape.n().unwrap_or(&1) as i32,
        ih: in_shape.hw_dims()[0] as i32,
        iw: in_shape.hw_dims()[1] as i32,
        c: *in_shape.c() as i32,
        oh: out_shape.hw_dims()[0] as i32,
        ow: out_shape.hw_dims()[1] as i32,
        kh: pool_spec.kernel_shape[0] as i32,
        kw: pool_spec.kernel_shape[1] as i32,
        stride_h: strides[0] as i32,
        stride_w: strides[1] as i32,
        pad_h: padding[0].pad_before as i32,
        pad_w: padding[1].pad_before as i32,
        dil_h: dilations[0] as i32,
        dil_w: dilations[1] as i32,
        count_include_pad: count_include_pad as i32,
        normalize: normalize as i32,
    };

    let channels_last = pool_spec.data_format.c_is_last();
    let base = match kind {
        PoolKind::Max if channels_last => "max_pool_2d",
        PoolKind::Max => "max_pool_2d_nchw",
        PoolKind::Sum { .. } => "sum_pool_2d",
    };
    ensure!(channels_last || kind == PoolKind::Max, "Metal sum pool is channels-last only");
    let pipeline = stream.load_pipeline(LibraryName::NNOps, &format!("{base}_{tname}"))?;

    stream.retain_tensor(input);
    stream.retain_tensor(output);

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(2, std::slice::from_ref(&params));
        let (fastest, height, depth) = if channels_last {
            (params.c, params.ow, params.oh * params.n)
        } else {
            (params.ow, params.oh, params.c * params.n)
        };
        let group_w = 32u64.min(fastest as u64).max(1);
        encoder.dispatch_threads(
            MTLSize { width: fastest as _, height: height as _, depth: depth as _ },
            MTLSize { width: group_w, height: 1, depth: 1 },
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalTransform;
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::ops::cnn::{MaxPool, PaddingSpec, SumPool};
    use tract_core::ops::nn::DataFormat;
    use tract_core::transform::ModelTransform;
    use tract_gpu::tensor::IntoDevice;

    fn ramp(dt: DatumType, shape: &[usize]) -> TractResult<Tensor> {
        let len: usize = shape.iter().product();
        let v: Vec<f32> = (0..len).map(|i| ((i * 17 % 61) as f32 - 30.0) / 8.0).collect();
        Ok(Tensor::from_shape(shape, &v)?.cast_to_dt(dt)?.into_owned())
    }

    fn spec(k: usize, stride: usize, padding: PaddingSpec, c: usize) -> PoolSpec {
        spec_fmt(DataFormat::NHWC, k, stride, padding, c)
    }

    fn spec_fmt(
        data_format: DataFormat,
        k: usize,
        stride: usize,
        padding: PaddingSpec,
        c: usize,
    ) -> PoolSpec {
        PoolSpec::new(data_format, tvec![k, k], padding, None, Some(tvec![stride, stride]), c, c)
    }

    /// Run the CPU op and the Metal kernel over the same input.
    fn check(
        dt: DatumType,
        shape: &[usize],
        pool_spec: PoolSpec,
        kind: PoolKind,
        cpu: Box<dyn TypedOp>,
    ) -> TractResult<()> {
        let input = ramp(dt, shape)?;
        let mut model = TypedModel::default();
        let i = model.add_source("i", dt.fact(shape))?;
        let out = model.wire_node("pool", cpu, &[i])?;
        model.select_output_outlets(&out)?;
        let expected = model
            .clone()
            .into_optimized()?
            .into_runnable()?
            .run(tvec![input.clone().into_tvalue()])?[0]
            .clone()
            .into_tensor();
        let got = with_borrowed_metal_stream(|stream| {
            let i = input.clone().into_device()?;
            let o_shape = pool_spec.output_shape(&input.shape().to_vec())?;
            let o = unsafe { DeviceTensor::uninitialized_dt(dt, &o_shape.shape)? };
            dispatch_metal_pool(stream, &pool_spec, kind, &i, &o)?;
            stream.wait_until_completed()?;
            Ok(o.to_host()?.into_tensor())
        })?;
        expected
            .close_enough(&got, Approximation::Approximate)
            .with_context(|| format!("{kind:?} dt={dt:?} shape={shape:?}"))
    }

    #[test]
    fn max_pool_3x3_stride2() -> TractResult<()> {
        let s = spec(3, 2, PaddingSpec::Valid, 32);
        check(
            DatumType::F32,
            &[1, 16, 16, 32],
            s.clone(),
            PoolKind::Max,
            Box::new(MaxPool::new(s, None)),
        )
    }

    #[test]
    fn max_pool_nchw() -> TractResult<()> {
        let s = spec_fmt(DataFormat::NCHW, 2, 2, PaddingSpec::Valid, 24);
        check(
            DatumType::F32,
            &[1, 24, 16, 16],
            s.clone(),
            PoolKind::Max,
            Box::new(MaxPool::new(s, None)),
        )
    }

    #[test]
    fn max_pool_nchw_same_padding() -> TractResult<()> {
        let s = spec_fmt(DataFormat::NCHW, 3, 2, PaddingSpec::SameUpper, 12);
        check(
            DatumType::F16,
            &[2, 12, 11, 13],
            s.clone(),
            PoolKind::Max,
            Box::new(MaxPool::new(s, None)),
        )
    }

    #[test]
    fn max_pool_same_padding() -> TractResult<()> {
        let s = spec(3, 1, PaddingSpec::SameUpper, 48);
        check(
            DatumType::F32,
            &[1, 13, 11, 48],
            s.clone(),
            PoolKind::Max,
            Box::new(MaxPool::new(s, None)),
        )
    }

    #[test]
    fn avg_pool_3x3_same() -> TractResult<()> {
        let s = spec(3, 1, PaddingSpec::SameUpper, 64);
        check(
            DatumType::F32,
            &[1, 12, 12, 64],
            s.clone(),
            PoolKind::Sum { count_include_pad: false, normalize: true },
            Box::new(SumPool::new(s, false, true)),
        )
    }

    #[test]
    fn avg_pool_count_include_pad() -> TractResult<()> {
        let s = spec(3, 2, PaddingSpec::SameUpper, 16);
        check(
            DatumType::F32,
            &[2, 9, 9, 16],
            s.clone(),
            PoolKind::Sum { count_include_pad: true, normalize: true },
            Box::new(SumPool::new(s, true, true)),
        )
    }

    #[test]
    fn sum_pool_no_normalize() -> TractResult<()> {
        let s = spec(2, 2, PaddingSpec::Valid, 32);
        check(
            DatumType::F32,
            &[1, 8, 8, 32],
            s.clone(),
            PoolKind::Sum { count_include_pad: false, normalize: false },
            Box::new(SumPool::new(s, false, false)),
        )
    }

    #[test]
    fn pool_f16() -> TractResult<()> {
        let s = spec(3, 2, PaddingSpec::SameUpper, 96);
        check(
            DatumType::F16,
            &[1, 14, 14, 96],
            s.clone(),
            PoolKind::Max,
            Box::new(MaxPool::new(s, None)),
        )
    }

    // The pools must actually land on the GPU once the transform has run.
    #[test]
    fn pools_route_through_metal_transform() -> TractResult<()> {
        let dt = DatumType::F32;
        let shape = [1usize, 14, 14, 32];
        let input = ramp(dt, &shape)?;
        let mut model = TypedModel::default();
        let i = model.add_source("i", dt.fact(&shape))?;
        let m = model.wire_node(
            "max",
            MaxPool::new(spec(3, 1, PaddingSpec::SameUpper, 32), None),
            &[i],
        )?[0];
        let a = model.wire_node(
            "avg",
            SumPool::new(spec(3, 1, PaddingSpec::SameUpper, 32), false, true),
            &[m],
        )?;
        model.select_output_outlets(&a)?;
        let cpu = model
            .clone()
            .into_optimized()?
            .into_runnable()?
            .run(tvec![input.clone().into_tvalue()])?;
        let metal = MetalTransform::default().transform_into(model.into_optimized()?)?;
        let n_pools =
            metal.nodes().iter().filter(|n| n.op_is::<crate::ops::pool::MetalPool>()).count();
        assert_eq!(n_pools, 2, "both pools should be on the GPU");
        let got = metal.into_runnable()?.run(tvec![input.into_tvalue()])?;
        cpu[0]
            .clone()
            .into_tensor()
            .close_enough(&got[0].clone().into_tensor(), Approximation::Approximate)?;
        Ok(())
    }
}

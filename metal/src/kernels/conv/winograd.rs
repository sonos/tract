use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_gpu::tensor::DeviceTensor;

/// Edge of the output tile one F(6x6, 3x3) transform produces.
pub const WINOGRAD_TILE: usize = 6;

/// Positions of the 8x8 transformed tile, the batch of the Winograd GEMM.
pub const WINOGRAD_POSITIONS: usize = 64;

/// The extents the Winograd kernels walk, mirrored by `WinogradGeometry` in
/// `winograd.metal`. Data is NHWC, the kernel OHWI, `pad_*` the padding before
/// each spatial axis.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WinogradGeometry {
    pub n: i32,
    pub h: i32,
    pub w: i32,
    pub c: i32,
    pub o: i32,
    pub oh: i32,
    pub ow: i32,
    pub pad_h: i32,
    pub pad_w: i32,
    pub tiles_h: i32,
    pub tiles_w: i32,
}

impl WinogradGeometry {
    /// The output tiles, the GEMM's rows.
    pub fn tiles(&self) -> usize {
        (self.n * self.tiles_h * self.tiles_w) as usize
    }

    /// The geometry of `conv` over `in_facts`, when F(6x6, 3x3) runs it: a
    /// plain 3x3 stride-1 undilated 2D conv, channels last, an `OHWI` kernel of
    /// the input's float type, and concrete shapes. As in mlx, the input has to
    /// be large enough, in pixels and channels, for the transforms to pay off.
    pub fn for_conv(conv: &Conv, in_facts: &[&TypedFact]) -> Option<WinogradGeometry> {
        let dt = in_facts[0].datum_type;
        if conv.group != 1
            || conv.q_params.is_some()
            || conv.kernel_fmt != KernelFormat::OHWI
            || !matches!(dt, DatumType::F16 | DatumType::F32)
            || in_facts[1].datum_type != dt
            || !conv.pool_spec.data_format.c_is_last()
            || conv.pool_spec.kernel_shape.as_slice() != [3, 3]
            || conv.pool_spec.strides().iter().any(|s| *s != 1)
            || conv.pool_spec.dilations().iter().any(|d| *d != 1)
        {
            return None;
        }
        let shape = in_facts[0].shape.as_concrete()?;
        let shape = conv.pool_spec.data_format.shape(shape).ok()?;
        let n = *shape.n().unwrap_or(&1);
        let (h, w) = (shape.hw_dims()[0], shape.hw_dims()[1]);
        let c = *shape.c();
        let o = conv.pool_spec.output_channels;
        if n * h * w < 4096 || c + o < 256 {
            return None;
        }
        let padding = conv.pool_spec.computed_padding(shape.hw_dims());
        let (oh, ow) = (padding[0].convoluted, padding[1].convoluted);
        let i = |x: usize| x as i32;
        Some(WinogradGeometry {
            n: i(n),
            h: i(h),
            w: i(w),
            c: i(c),
            o: i(o),
            oh: i(oh),
            ow: i(ow),
            pad_h: i(padding[0].pad_before),
            pad_w: i(padding[1].pad_before),
            tiles_h: i(oh.div_ceil(WINOGRAD_TILE)),
            tiles_w: i(ow.div_ceil(WINOGRAD_TILE)),
        })
    }
}

/// One of the three transforms around the Winograd GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WinogradStage {
    /// `OHWI` kernel to `U[64, C, O]`.
    Weights,
    /// NHWC data to `V[64, T, C]`.
    Input,
    /// `M[64, T, O]` to NHWC output.
    Output,
}

impl WinogradStage {
    fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            _ => bail!("Winograd conv: F32/F16 only, got {dt:?}"),
        };
        let stage = match self {
            Self::Weights => "weights",
            Self::Input => "input",
            Self::Output => "output",
        };
        Ok(format!("winograd_{stage}_{tname}"))
    }

    /// The threads a stage runs: the inner channel axis along `x`, so that
    /// both its reads and writes run contiguously across a simdgroup.
    fn grid(&self, geo: &WinogradGeometry) -> (usize, usize) {
        match self {
            Self::Weights => (geo.o as usize, geo.c as usize),
            Self::Input => (geo.c as usize, geo.tiles()),
            Self::Output => (geo.o as usize, geo.tiles()),
        }
    }
}

pub fn dispatch_winograd_stage(
    stream: &MetalStream,
    stage: WinogradStage,
    geo: &WinogradGeometry,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let typed = if stage == WinogradStage::Output { output } else { input };
    let pipeline =
        stream.load_pipeline(LibraryName::Winograd, &stage.kernel_name(typed.datum_type())?)?;
    stream.retain_tensor(input);
    stream.retain_tensor(output);
    let (x, y) = stage.grid(geo);
    let group = MTLSize { width: 32, height: 8, depth: 1 };
    stream.command_buffer().encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(2, std::slice::from_ref(geo));
        encoder.dispatch_thread_groups(
            MTLSize {
                width: x.div_ceil(group.width as usize) as _,
                height: y.div_ceil(group.height as usize) as _,
                depth: 1,
            },
            group,
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::MetalTransform;
    use crate::ops::winograd::MetalWinograd;
    use tract_core::internal::*;
    use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    use tract_core::transform::ModelTransform;

    fn routes_through_winograd(dt: DatumType, hw: usize, c: usize, o: usize) -> TractResult<bool> {
        let pool_spec = PoolSpec::new(
            DataFormat::NHWC,
            tvec![3, 3],
            PaddingSpec::SameUpper,
            Some(tvec![1, 1]),
            Some(tvec![1, 1]),
            c,
            o,
        );
        let op = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };
        let mut model = TypedModel::default();
        let i = model.add_source("i", dt.fact([1, hw, hw, c]))?;
        let kernel = Tensor::zero_dt(dt, &[o, c, 3, 3])?;
        let w = model.add_const("w", kernel)?;
        let b = model.add_const("b", Tensor::zero_dt(dt, &[o])?)?;
        let out = model.wire_node("conv", op, &[i, w, b])?;
        model.select_output_outlets(&out)?;
        let metal = MetalTransform::default().transform_into(model)?;
        Ok(metal.nodes().iter().any(|n| n.op_is::<MetalWinograd>()))
    }

    /// The suite-unit `winograd_*` conv cases only see the result, the same
    /// either way, so the routing they are sized for is checked here.
    #[test]
    fn winograd_takes_large_3x3_convs() -> TractResult<()> {
        for dt in [DatumType::F16, DatumType::F32] {
            assert!(routes_through_winograd(dt, 64, 128, 128)?, "{dt:?}");
            assert!(!routes_through_winograd(dt, 12, 128, 128)?, "{dt:?}: too few pixels");
            assert!(!routes_through_winograd(dt, 64, 64, 64)?, "{dt:?}: too few channels");
        }
        Ok(())
    }
}

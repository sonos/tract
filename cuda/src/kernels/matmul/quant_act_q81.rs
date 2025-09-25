use cudarc::driver::{LaunchConfig, PushKernelArg};
use derive_new::new;
use std::fmt;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q8_1};
use tract_gpu::tensor::DeviceTensor;

use crate::Q40_ROW_PADDING;
use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::matmul::{MMQ_X_MAX, squeeze_batch_axes};
use crate::kernels::{LibraryName, get_cuda_view};
use crate::ops::GgmlQuantQ81Fact;

pub(crate) const QK8_1: usize = 32;

pub(crate) const QUANTIZE_BLOCK_SIZE: usize = 256;
pub(crate) const QUANTIZE_BLOCK_SIZE_MMQ: usize = 128;

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct GgmlQuantQ81;

impl fmt::Display for GgmlQuantQ81 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl GgmlQuantQ81 {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn output_shape_fact(shape: &ShapeFact) -> TractResult<ShapeFact> {
        let mut o_shape = shape.dims().to_owned();
        let rank = o_shape.len();
        let k = o_shape[rank - 1].as_i64().context("Expected concrete k")? as usize;
        let padded_k = k.next_multiple_of(Q40_ROW_PADDING);
        o_shape[rank - 1] = TDim::Val(padded_k as i64);
        o_shape[rank - 2] += (MMQ_X_MAX * 4 * Q8_1.block_bytes()).div_ceil(padded_k);

        Ok(ShapeFact::from_dims(o_shape))
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        let context = cuda_context();
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let rank = input.rank();
        let squeezed_shape = squeeze_batch_axes(input.shape())?;

        let a_batch = squeezed_shape[0];
        let m = squeezed_shape[1];
        let k = squeezed_shape[2];

        let padded_k = k.next_multiple_of(Q40_ROW_PADDING);
        let mut out_shape = input.shape().to_owned();
        out_shape[rank - 1] = padded_k;
        if m > 8 {
            let in_strides = input.strides();
            let fast_path_str = if in_strides[rank - 1] == 1 { "fast_" } else { "" };
            let func = cuda_context().load_pipeline(
                LibraryName::Quant,
                format!("quantize_mmq_q8_1_{fast_path_str}nd{}", input.rank()),
            )?;
            let mut launch_args = stream.launch_builder(&func);
            launch_args.arg(&i_view);
            launch_args.arg(&o_view);
            launch_args.arg(&k);
            launch_args.set_slice(in_strides);
            launch_args.set_slice(&out_shape[1..]);

            let cfg = LaunchConfig {
                grid_dim: (
                    m as _,
                    padded_k.div_ceil(4 * QUANTIZE_BLOCK_SIZE_MMQ) as _,
                    a_batch as _,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { launch_args.launch(cfg) };
        } else {
            let func = context
                .load_pipeline(LibraryName::Quant, format!("quantize_q8_1_nd{}", input.rank()))?;
            let mut launch_args = stream.launch_builder(&func);
            launch_args.arg(&i_view);
            launch_args.arg(&o_view);
            launch_args.arg(&k);
            launch_args.set_slice(input.strides());
            launch_args.set_slice(&out_shape[1..]);

            let cfg = LaunchConfig {
                grid_dim: (padded_k.div_ceil(QUANTIZE_BLOCK_SIZE) as _, m as _, a_batch as _),
                block_dim: (QUANTIZE_BLOCK_SIZE as _, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { launch_args.launch(cfg) };
        }
        Ok(())
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output_fact: GgmlQuantQ81Fact,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_opaque(Box::new(output_fact))? };
        self.dispatch_eval(stream, input, &output)?;
        stream.synchronize()?;
        Ok(output)
    }
}

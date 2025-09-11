use std::fmt;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use derive_new::new;
use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0, Q8_1};
use tract_gpu::tensor::DeviceTensor;

use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::matmul::ggml::{squeeze_batch_axes, MMQ_X_MAX};
use crate::kernels::{get_cuda_view, get_sliced_cuda_view, LibraryName};
use crate::Q40_ROW_PADDING;

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

    pub fn output_shape(shape: &[usize], mmq: bool) -> TractResult<TVec<usize>> {
        let mut o_shape: TVec<usize> = shape.to_smallvec();
        let rank = o_shape.len();
        let k = o_shape[rank - 1].next_multiple_of(Q40_ROW_PADDING);
        o_shape[rank - 1] = k;
        if mmq {
            o_shape[rank - 2] += (MMQ_X_MAX * 4 * Q8_1.block_bytes()).div_ceil(k);
        }
        Ok(o_shape)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
        mmq: bool,
    ) -> TractResult<()> {
        let context = cuda_context();
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let squeezed_shape = squeeze_batch_axes(input.shape())?;

        let a_batch = squeezed_shape[0];
        let m = squeezed_shape[1];
        let k = squeezed_shape[2];

        let a_strides = natural_strides(&[a_batch, m, k]);
        let sample_stride_a = a_strides[0] * a_batch as isize;

        let padded_k = k.next_multiple_of(Q40_ROW_PADDING);

        if mmq {
            let func = cuda_context().load_pipeline(LibraryName::GgmlQ, "quantize_mmq_q8_1".to_string())?;
            let mut launch_args = stream.launch_builder(&func);
            launch_args.arg(&i_view);
            launch_args.arg(&o_view);
            launch_args.arg(&k);
            launch_args.arg(&a_strides[1]);
            launch_args.arg(&a_strides[0]);
            launch_args.arg(&sample_stride_a);
            launch_args.arg(&padded_k);
            launch_args.arg(&m);
            launch_args.arg(&a_batch);

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
            let func = context.load_pipeline(LibraryName::GgmlQ, "quantize_q8_1".to_string())?;
            let mut launch_args = stream.launch_builder(&func);
            launch_args.arg(&i_view);
            launch_args.arg(&o_view);
            launch_args.arg(&k);
            launch_args.arg(&a_strides[1]);
            launch_args.arg(&a_strides[0]);
            launch_args.arg(&sample_stride_a);
            launch_args.arg(&padded_k);
            launch_args.arg(&m);
            launch_args.arg(&a_batch);

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
        output_shape: &[usize],
        mmq: bool,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), output_shape)? };
        self.dispatch_eval(stream, input, &output, mmq)?;
        stream.synchronize()?;
        Ok(output)
    }
}
use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{get_cuda_view, get_sliced_cuda_view, utils, BroadcastKind, LibraryName};
use anyhow::ensure;
use cudarc::driver::{CudaStream, PushKernelArg};
use std::fmt;
use std::ops::Range;
use std::ops::RangeBounds;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Concat {
    pub axis: usize,
}

impl fmt::Display for Concat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Concat {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
            dt,
            DatumType::F32
                | DatumType::F16
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64
                | DatumType::I8
                | DatumType::I16
                | DatumType::I32
                | DatumType::I64
        )
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda concatop", dt);
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("copy_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        inputs: &[&DeviceTensor],
    ) -> TractResult<DeviceTensor> {
        ensure!(!inputs.is_empty());
        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[self.axis] = inputs.iter().map(|it| it.shape()[self.axis]).sum();
        let output =
            unsafe { DeviceTensor::uninitialized_dt(inputs[0].datum_type(), &output_shape)? };

        self.dispatch_eval(stream, inputs, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        inputs: &[&DeviceTensor],
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(!inputs.is_empty());
        let mut cursor = 0;
        for (i_idx, input) in inputs.iter().enumerate() {
            let slice_len = input.shape()[self.axis];

            device_tensor_assign_slice(
                stream,
                output,
                cursor..cursor + slice_len,
                input,
                ..,
                self.axis,
            )?;
            cursor += slice_len;
        }

        Ok(())
    }
}

fn device_tensor_assign_slice(
    stream: &TractCudaStream,
    dst: &DeviceTensor,
    dst_range: impl RangeBounds<usize>,
    src: &DeviceTensor,
    src_range: impl RangeBounds<usize>,
    axis: usize,
) -> TractResult<()> {
    ensure!(src.datum_type() == dst.datum_type());
    ensure!(src.datum_type().is_copy() && src.datum_type().is_number());
    ensure!(src.rank() == dst.rank() && axis < src.rank());
    let src_range = clip_range_bounds(src.shape()[axis], src_range);
    let dst_range = clip_range_bounds(dst.shape()[axis], dst_range);
    ensure!(dst_range.len() == src_range.len());
    ensure!(
        tract_itertools::izip!(dst.shape(), src.shape(), 0..).all(|(d, s, a)| a == axis || s == d)
    );

    let broadcast_kind = BroadcastKind::from_rank(dst.rank()).with_context(|| {
        format!(
            "Unsupported broadcast for assign slice: (in: {:?}, out: {:?})",
            src.shape(),
            dst.shape()
        )
    })?;

    let tname = DeviceTensor::tname(src.datum_type())?;
    let broadcast_name = broadcast_kind.name();
    let kernel_name = format!("copy_{broadcast_name}_{tname}");
    let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;

    let mut shape = src.shape().to_vec();
    shape[axis] = src_range.len();

    let src_offset = src.strides()[axis] as usize * src_range.start * src.datum_type().size_of();
    let src_bytes = src.len() * src.datum_type().size_of();
    let src_view = get_sliced_cuda_view(src, src_offset, src_bytes - src_offset)?;

    let dst_offset = dst.strides()[axis] as usize * dst_range.start * dst.datum_type().size_of();
    let dst_bytes = dst.len() * dst.datum_type().size_of();
    let dst_view = get_sliced_cuda_view(dst, dst_offset, dst_bytes - dst_offset)?;

    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(&src_view);
    launch_args.arg(&dst_view);
    launch_args.set_slice(src.strides());
    launch_args.set_slice(&shape);
    launch_args.set_slice(dst.strides());

    let cfg = utils::cuda_launch_cfg_for_cpy(&shape);

    unsafe { launch_args.launch(cfg) };

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::context::CUDA_STREAM;

    use super::*;
    use tract_gpu::tensor::IntoDevice;
    use tract_itertools::Itertools;

    use num_traits::Zero;

    use tract_core::internal::Tensor;

    fn run_test_case(shapes: &[&[usize]], axis: usize) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let mut inputs = tvec![];
            for shape in shapes {
                let len = shape.iter().product::<usize>();
                let data = (0..len).map(|f| f as f32).collect::<Vec<_>>();
                inputs.push(Tensor::from_shape(shape, &data)?.into_device()?);
            }

            let output = Concat { axis }.eval(stream, &inputs.iter().collect_vec())?;
            let ref_output = Tensor::stack_tensors(
                axis,
                &inputs.iter().map(|it| it.to_host()).collect::<TractResult<Vec<_>>>()?,
            )?;
            assert_eq!(output.to_host()?.into_tensor(), ref_output);
            Ok(())
        })
    }

    #[test]
    fn test_concat() -> TractResult<()> {
        run_test_case(&[&[3, 4], &[3, 4]], 0)?;
        run_test_case(&[&[3, 4], &[3, 4]], 1)?;
        run_test_case(&[&[1, 5, 4], &[2, 5, 4]], 0)?;
        run_test_case(&[&[3, 1, 8], &[3, 2, 8]], 1)?;
        run_test_case(&[&[3, 5, 1], &[3, 5, 2]], 2)?;
        run_test_case(&[&[1, 5, 4, 10], &[2, 5, 4, 10]], 0)?;
        run_test_case(&[&[3, 1, 8, 10], &[3, 2, 8, 10]], 1)?;
        run_test_case(&[&[3, 5, 1, 10], &[3, 5, 2, 10]], 2)?;
        run_test_case(&[&[3, 5, 1, 10], &[3, 5, 0, 10]], 2)?;
        run_test_case(&[&[3, 5, 1, 2], &[3, 5, 1, 8]], 3)?;
        Ok(())
    }
}

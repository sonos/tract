use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{get_cuda_view, get_sliced_cuda_view, utils, BroadcastKind, LibraryName};
use crate::tensor::device_tensor_assign_slice;
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
            for (ix, shape) in shapes.iter().enumerate() {
                let len = shape.iter().product::<usize>();
                let data = (0..len).map(|f| (100 * ix + f) as f32).collect::<Vec<_>>();
                inputs.push(Tensor::from_shape(shape, &data)?.into_device()?);
            }

            let output = Concat { axis }.eval(stream, &inputs.iter().collect_vec())?;
            let ref_output = Tensor::stack_tensors(
                axis,
                &inputs.iter().map(|it| it.to_host()).collect::<TractResult<Vec<_>>>()?,
            )?;
            let output = output.to_host()?.into_tensor();
            if output != ref_output {
                eprintln!("AXIS: {axis}");
                for (ix, input) in inputs.iter().enumerate() {
                    eprintln!("INPUT {ix}:\n{}", input.to_host().unwrap().to_array_view::<f32>()?);
                }
                eprintln!("FOUND:\n{}", output.to_array_view::<f32>()?);
                eprintln!("EXPECTED:\n{}", ref_output.to_array_view::<f32>()?);
                panic!("Test failed");
            }
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

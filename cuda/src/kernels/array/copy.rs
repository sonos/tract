use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use derive_new::new;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::cuda_context;
use crate::kernels::{get_cuda_view, get_sliced_cuda_view, LibraryName};

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct Memcpy;

impl fmt::Display for Memcpy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Memcpy {
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
                | DatumType::Bool
        )
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for cuda copy  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("copy_unicast_{tname}"))
    }

    pub fn dispatch_eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        input_offset: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input_offset % input.datum_type().size_of() == 0);
        ensure!(output.len() <= input.len() - (input_offset / input.datum_type().size_of()));

        let kernel_name = self.kernel_name(input.datum_type())?;

        let func = cuda_context().load_pipeline(LibraryName::ArrayOps, kernel_name)?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);

        let cfg = LaunchConfig::for_num_elems(output.len() as _);
        unsafe { launch_args.launch(cfg) };
        Ok(())
    }

    pub fn eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        input_offset: usize,
        output_shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), output_shape)? };
        self.dispatch_eval(stream, input, input_offset, &output)?;
        stream.synchronize()?;
        Ok(output)
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

    fn run_test_case(shape: &[usize], offset: usize) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let len = shape.iter().product::<usize>();
            let data = (0..len).map(|f| f as f32).collect::<Vec<_>>();
            let input = Tensor::from_shape(shape, &data)?;

            let output = Memcpy{}.eval(stream, &input.clone().into_device()?, offset, &[len - (offset / size_of::<f32>())])?;

            assert_eq!(output.to_host()?.into_tensor(), 
                        input.into_shape(&[len])?.slice(0, offset / size_of::<f32>(), len)? );
            Ok(())
        })
    }

    #[test]
    fn test_cpy() -> TractResult<()> {
        run_test_case(&[3, 4], 0)?;
        run_test_case(&[2, 5], 2 * size_of::<f32>())?;
        Ok(())
    }
}
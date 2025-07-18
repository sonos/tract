use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use derive_new::new;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::{LibraryName, get_cuda_view, launch_args};

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct Cast;

impl fmt::Display for Cast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Cast {
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

    pub fn kernel_name(&self, from_dt: DatumType, to_dt: DatumType) -> TractResult<String> {
        ensure!(
            Self::is_supported_dt(from_dt),
            "Unsupported from_dt {:?} for cuda castop",
            from_dt
        );
        ensure!(Self::is_supported_dt(to_dt), "Unsupported to_dt {:?} for cuda castop", to_dt);
        let from_tname = DeviceTensor::tname(from_dt)?;
        let to_tname = DeviceTensor::tname(to_dt)?;
        Ok(format!("cast_{from_tname}_{to_tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        to_dt: DatumType,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(to_dt, input.shape())? };
        self.dispatch_eval(stream, input, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(
            input.shape() == output.shape(),
            "Cast I/O don't have the same shape in: {:?}, out: {:?}",
            input.shape(),
            output.shape()
        );

        let kernel_name = self.kernel_name(input.datum_type(), output.datum_type())?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        let len = output.len();
        let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;

        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.arg(&len);
        let cfg = LaunchConfig::for_num_elems(len as _);

        unsafe { launch_args.launch(cfg) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::context::CUDA_STREAM;

    use super::*;
    use tract_gpu::tensor::IntoDevice;
    use tract_itertools::Itertools;

    use num_traits::{FromPrimitive, Zero};

    use tract_core::internal::Tensor;

    fn run_test_case<T0: Datum + Copy + FromPrimitive, T1: Datum>(
        shape: &[usize],
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let len = shape.iter().product::<usize>();
            let data = (0..len).map(|f| T0::from_f32(f as f32 / 2.).unwrap()).collect::<Vec<_>>();
            let input = Tensor::from_shape(shape, &data)?;

            let output = Cast {}.eval(stream, &input.clone().into_device()?, T1::datum_type())?;

            assert_eq!(
                output.to_host()?.into_tensor(),
                input.cast_to_dt(T1::datum_type())?.into_owned()
            );
            Ok(())
        })
    }

    #[test]
    fn test_cast() -> TractResult<()> {
        run_test_case::<f16, f32>(&[3, 4])?;
        run_test_case::<u8, f32>(&[2, 5])?;
        run_test_case::<f16, u32>(&[3, 2, 2])?;
        Ok(())
    }
}

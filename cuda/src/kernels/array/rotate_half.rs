use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::{LibraryName, get_cuda_view, utils};
use anyhow::ensure;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RotateHalf;

impl fmt::Display for RotateHalf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl RotateHalf {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
            dt,
            DatumType::F32
                | DatumType::F16
                | DatumType::I8
                | DatumType::I16
                | DatumType::I32
                | DatumType::I64
        )
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda rotate halfop", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("rotate_half_nd2_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
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
        let shape_nd2 = utils::reshape_to_rank_2(input.shape(), input.rank() - 1);
        ensure!(
            shape_nd2[1] % 2 == 0,
            "Rotate half required most inner dimension to be a multiple of 2: {:?}",
            input.shape()
        );
        let strides_nd2 = Tensor::natural_strides(&shape_nd2);

        let kernel_name = self.kernel_name(input.datum_type())?;

        let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let mut launch_args = stream.launch_builder(&func);
        launch_args.set_view(&i_view);
        launch_args.set_view(&o_view);
        launch_args.set_slice::<i64>(&shape_nd2);
        launch_args.set_slice::<i64>(&strides_nd2);

        let cfg = LaunchConfig {
            grid_dim: ((shape_nd2[1] / 2) as _, shape_nd2[0] as _, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_args.launch(cfg)
    }
}

#[cfg(test)]
mod tests {
    use crate::context::CUDA_STREAM;

    use super::*;
    use num_traits::AsPrimitive;
    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::apply_rope;

    fn run_test_case<F>(shape: &[usize]) -> TractResult<()>
    where
        F: Copy + 'static + Datum,
        usize: AsPrimitive<F>,
    {
        CUDA_STREAM.with(|stream| {
            let len = shape.iter().product::<usize>();

            let a =
                Tensor::from_shape(shape, &(0..len).map(|f| -> F { f.as_() }).collect::<Vec<_>>())?;

            let cuda_a = a.clone().into_device()?;

            let cpu_output =
                apply_rope::RotateHalf.eval(tvec![a.clone().into()])?[0].clone().into_tensor();
            let cuda_output = RotateHalf.eval(stream, &cuda_a)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Exact)
                .with_context(|| {
                    format!(
                        "Input: {:?} Cpu: {:?}, Cuda: {:?}",
                        a.dump(true),
                        cpu_output.dump(true),
                        cuda_output.to_host().and_then(|it| it.dump(true))
                    )
                })?;
            Ok(())
        })
    }

    #[test]
    fn test_rotate_half() -> TractResult<()> {
        run_test_case::<f32>(&[2, 2])?;
        run_test_case::<f32>(&[512, 512])?;
        run_test_case::<f32>(&[10, 8, 8])?;
        run_test_case::<f32>(&[10, 512, 1024])?;
        run_test_case::<f32>(&[10, 512, 1024])?;
        run_test_case::<i8>(&[10, 1, 2])?;
        run_test_case::<i16>(&[10, 256, 4])?;
        run_test_case::<i32>(&[10, 512, 1024])?;
        Ok(())
    }
}

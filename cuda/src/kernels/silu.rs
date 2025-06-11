use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::cuda_context;
use crate::kernels::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Silu;

impl Silu {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for Cuda silu op", dt);
        Ok(format!("unary_{}_{}", "silu", DeviceTensor::tname(dt)?))
    }

    #[allow(unused)]
    pub fn eval(&self, stream: &CudaStream, input: &DeviceTensor) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, &output)?;
        stream.synchronize()?;

        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let func =
            cuda_context().load_pipeline(LibraryName::UnaryOps, self.kernel_name(input.datum_type())?)?;

        let len = input.len();

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let cfg = LaunchConfig::for_num_elems(len as _);
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.arg(&len);

        unsafe { launch_args.launch(cfg) }?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{AsPrimitive, Float};
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::silu;

    use crate::context::{CUDA_STREAM, cuda_context};

    fn test_case<F>(
        shape: &[usize],
        offset: f32,
        scale: f32,
        appriximate: Approximation,
    ) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        cuda_context();
        CUDA_STREAM.with(|stream| {
            let len = shape.iter().product::<usize>();

            let a = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v * scale + offset).as_()
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cpu_output =
                silu::Silu.eval(tvec![a.to_host()?.into_tvalue()])?[0].clone().into_tensor();
            let cuda_output = Silu.eval(stream, &a)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), appriximate)
                .with_context(|| {
                    format!(
                        "Input: {:?}, scale: {:?} Cpu: {:?}, CUDA: {:?}",
                        a.to_host().and_then(|it| it.dump(true)),
                        scale,
                        cpu_output.dump(true),
                        cuda_output.to_host().and_then(|it| it.dump(true))
                    )
                })?;
            Ok(())
        })
    }

    #[test]
    fn test_silu() -> TractResult<()> {
        test_case::<f32>(&[4, 4], -0.0, 1.0 / 100.0, Approximation::Approximate)?;
        test_case::<f16>(&[4, 4], -6.0, 1.0 / 1000.0, Approximation::VeryApproximate)?;
        Ok(())
    }
}

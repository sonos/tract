use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::{LibraryName, get_cuda_view};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeakyRelu;

impl LeakyRelu {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda geluop", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("leaky_relu_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        alpha: f32,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, alpha, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        alpha: f32,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let kernel_name = self.kernel_name(input.datum_type())?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        let len = output.len();

        let func = cuda_context().load_pipeline(LibraryName::NN, kernel_name)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.arg(&len);
        launch_args.arg(&alpha);

        let cfg = LaunchConfig::for_num_elems(input.len() as _);
        unsafe {
            launch_args.launch(cfg);
        }
        Ok(())
    }
}

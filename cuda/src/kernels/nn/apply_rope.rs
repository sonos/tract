use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{BroadcastKind, LibraryName, get_cuda_view, utils};
use anyhow::ensure;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ApplyRope;

impl fmt::Display for ApplyRope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ApplyRope {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn is_supported_broadcast(broadcast_kind: BroadcastKind) -> bool {
        matches!(broadcast_kind, BroadcastKind::Nd2 | BroadcastKind::Nd3 | BroadcastKind::Nd4)
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda apply rope", dt);
        ensure!(
            Self::is_supported_broadcast(broadcast_kind),
            "Unsupported broadcast kind {:?} for cuda apply rope",
            broadcast_kind
        );
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("apply_rope_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        cos: &DeviceTensor,
        sin: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, cos, sin, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        cos: &DeviceTensor,
        sin: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input.datum_type() == cos.datum_type());
        ensure!(input.datum_type() == sin.datum_type());

        ensure!(cos.shape() == sin.shape());

        ensure!(input.rank() >= 2 && input.rank() <= 4);
        ensure!(cos.rank() <= input.rank());

        let padded_shape = [&tvec![1; input.rank() - cos.rank()], cos.shape()].concat();
        let (padded_cos, padded_sin) =
            (cos.reshaped(padded_shape.clone().into())?, sin.reshaped(padded_shape.into())?);

        ensure!(
            input.shape()[input.rank() - 1] % 2 == 0,
            "Rotate half required most inner dimension to be a multiple of 2: {:?}",
            input.shape()
        );

        let cos_sin_strides =
            compute_broadcast_strides::<usize>(padded_cos.shape(), padded_sin.strides())?;

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| format!("Unsupported rank for ApplyRope op: {:?}", input.shape(),))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let i_view = get_cuda_view(input);
        let cos_view = get_cuda_view(&padded_cos);
        let sin_view = get_cuda_view(&padded_sin);
        let o_view = get_cuda_view(output);

        let func = cuda_context().load_pipeline(LibraryName::NN, kernel_name)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&cos_view);
        launch_args.arg(&sin_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(input.shape());
        launch_args.set_slice(input.strides());
        launch_args.set_slice(&cos_sin_strides);
        launch_args.set_slice(output.strides());

        let shape = input.shape();

        let block_dim = 32;
        let mut grid = match shape.len() {
            0 => panic!("Unexpected empty shape while build grid size"),
            1 => (shape[0] as _, 1, 1),
            2 => (shape[1] as _, shape[0] as _, 1),
            3.. => (
                shape[shape.len() - 1],
                shape[shape.len() - 2],
                (shape[..shape.len() - 2].iter().product::<usize>()),
            ),
        };
        grid.0 /= 2;

        let cfg = LaunchConfig {
            grid_dim: (
                grid.0.div_ceil(block_dim) as _,
                grid.1.div_ceil(block_dim) as _,
                grid.2 as _,
            ),
            block_dim: (block_dim as _, block_dim as _, 1),
            shared_mem_bytes: 0,
        };

        unsafe { launch_args.launch(cfg) };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::context::CUDA_STREAM;

    use super::*;
    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::apply_rope;

    fn run_test_case(shape: &[usize]) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let len = shape.iter().product::<usize>();

            let a = Tensor::from_shape(
                shape,
                &(0..len).map(|f| f as f32 / 1000.0).collect::<Vec<_>>(),
            )?;

            let cos =
                Tensor::from_shape(shape, &(0..len).map(|f| (f as f32).cos()).collect::<Vec<_>>())?;

            let sin =
                Tensor::from_shape(shape, &(0..len).map(|f| (f as f32).sin()).collect::<Vec<_>>())?;

            let cuda_a = a.clone().into_device()?;
            let cuda_sin = sin.clone().into_device()?;
            let cuda_cos = cos.clone().into_device()?;

            let cpu_output = apply_rope::ApplyRope.eval(tvec![
                a.clone().into(),
                cos.clone().into(),
                sin.clone().into(),
            ])?[0]
                .clone()
                .into_tensor();
            let cuda_output = ApplyRope.eval(stream, &cuda_a, &cuda_cos, &cuda_sin)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)
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
    fn test_apply_rope() -> TractResult<()> {
        run_test_case(&[2, 1, 2, 2])?;
        run_test_case(&[2, 4, 4])?;
        run_test_case(&[2, 1, 512, 10])?;
        run_test_case(&[8, 8])?;
        run_test_case(&[1, 10, 512, 24])?;
        run_test_case(&[3, 10, 512, 24])?;
        Ok(())
    }
}

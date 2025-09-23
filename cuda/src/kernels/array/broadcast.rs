use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{BroadcastKind, LibraryName, get_cuda_view, get_sliced_cuda_view, utils};

use crate::kernels::utils::compute_broadcast_strides;
use anyhow::ensure;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiBroadcast;

impl fmt::Display for MultiBroadcast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl MultiBroadcast {
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

    pub fn is_supported_broadcast(broadcast_kind: BroadcastKind) -> bool {
        matches!(
            broadcast_kind,
            BroadcastKind::Nd1
                | BroadcastKind::Nd2
                | BroadcastKind::Nd3
                | BroadcastKind::Nd4
                | BroadcastKind::Nd5
                | BroadcastKind::Nd6
        )
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda broadcast op", dt);
        ensure!(
            Self::is_supported_broadcast(broadcast_kind),
            "Unsupported broadcast kind {:?} for cuda broadcast op",
            broadcast_kind
        );
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("copy_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        input_offset: usize,
        output_shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), output_shape)? };
        self.dispatch_eval(stream, input, input_offset, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        input_offset: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input.rank() <= output.rank(), "Input must have a rank lower or equal to output");

        let mut input_shape = vec![1; output.rank() - input.rank()];
        input_shape.extend(input.shape());
        let mut input_strides = vec![input.strides()[0]; output.rank() - input.rank()];
        input_strides.extend(input.strides());

        let broadcast_kind = BroadcastKind::from_rank(output.rank()).with_context(|| {
            format!(
                "Unsupported broadcast for broadcast op: (in: {:?}, out: {:?})",
                input.shape(),
                output.shape(),
            )
        })?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let input_broadcast_strides =
            compute_broadcast_strides::<usize>(input_shape.as_slice(), input_strides.as_slice())?;

        let out_shape = output.shape();

        let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;

        let i_view = get_sliced_cuda_view(
            input,
            input_offset,
            input.len() * input.datum_type().size_of() - input_offset,
        )?;
        let o_view = get_cuda_view(output);
        let mut launch_args = stream.launch_builder(&func);

        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&input_broadcast_strides);
        launch_args.set_slice(out_shape);
        launch_args.set_slice(output.strides());

        let cfg = utils::cuda_launch_cfg_for_cpy(out_shape);

        unsafe { launch_args.launch(cfg) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tract_gpu::tensor::IntoDevice;

    use super::*;

    use crate::context::CUDA_STREAM;

    fn run_test_case(in_shape: &[usize], out_shape: &[usize]) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_len = in_shape.iter().product::<usize>();

            let a =
                Tensor::from_shape(in_shape, &(0..a_len).map(|f| f as f32).collect::<Vec<_>>())?;
            let output = MultiBroadcast {}.eval(stream, &a.clone().into_device()?, 0, out_shape)?;
            let ref_output = a.broadcast_to_shape(out_shape)?;

            assert_eq!(output.to_host()?.into_tensor(), ref_output);
            Ok(())
        })
    }

    #[test]
    fn test_broadcast_nd1() -> TractResult<()> {
        run_test_case(&[2], &[2])?;
        run_test_case(&[1], &[4])?;
        Ok(())
    }

    #[test]
    fn test_broadcast_nd2() -> TractResult<()> {
        run_test_case(&[2, 4], &[2, 4])?;
        run_test_case(&[1, 2], &[4, 2])?;
        run_test_case(&[3, 1], &[3, 3])?;
        Ok(())
    }

    #[test]
    fn test_broadcast_nd3() -> TractResult<()> {
        run_test_case(&[2, 3, 4], &[2, 3, 4])?;
        run_test_case(&[1, 5, 2], &[4, 5, 2])?;
        run_test_case(&[3, 2, 1], &[3, 2, 3])?;
        Ok(())
    }

    #[test]
    fn test_broadcast_nd4() -> TractResult<()> {
        run_test_case(&[2, 3, 4, 5], &[2, 3, 4, 5])?;
        run_test_case(&[1, 5, 2, 3], &[4, 5, 2, 3])?;
        run_test_case(&[3, 2, 6, 1], &[3, 2, 6, 3])?;
        Ok(())
    }

    #[test]
    fn test_broadcast_nd5() -> TractResult<()> {
        run_test_case(&[2, 3, 1, 4, 5], &[2, 3, 1, 4, 5])?;
        run_test_case(&[1, 5, 1, 2, 3], &[4, 5, 5, 2, 3])?;
        run_test_case(&[3, 1, 2, 6, 1], &[3, 4, 2, 6, 3])?;
        Ok(())
    }

    #[test]
    fn test_broadcast_nd6() -> TractResult<()> {
        run_test_case(&[2, 3, 5, 2, 4, 5], &[2, 3, 5, 2, 4, 5])?;
        run_test_case(&[1, 5, 1, 1, 2, 3], &[4, 5, 3, 4, 2, 3])?;
        run_test_case(&[3, 1, 2, 6, 1, 1], &[3, 3, 2, 6, 2, 3])?;
        Ok(())
    }
}

use crate::context::cuda_context;
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{get_cuda_view, utils, BroadcastKind, LibraryName};
use std::fmt;
use cudarc::driver::{CudaStream, LaunchArgs, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PermuteAxes;

impl fmt::Display for PermuteAxes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl PermuteAxes {
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

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for cuda permute axes  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("copy_{broadcast_name}_{tname}"))
    }

    pub fn output_shape<D: DimLike>(shape: &[D], axes: &[usize]) -> TractResult<TVec<D>> {
        let rank = shape.len();
        let mut new_shape = tvec![0.into(); rank];

        for (new_axis, &axis) in axes.iter().enumerate() {
            new_shape[new_axis] = shape[axis].clone();
        }
        Ok(new_shape)
    }

    pub fn eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        axes: &[usize],
    ) -> TractResult<DeviceTensor> {
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                input.datum_type(),
                &Self::output_shape(input.shape(), axes)?,
            )?
        };
        self.dispatch_eval(stream, input, axes, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        axes: &[usize],
        output: &DeviceTensor,
    ) -> TractResult<()> {
        // Validate give axes permutation
        let mut usage_counts = vec![0; input.rank()];
        for axis in axes {
            usage_counts[*axis] += 1;
        }
        for count in usage_counts {
            assert_eq!(count, 1, "each axis must be listed exactly once");
        }

        let shape = input.shape();
        let strides = input.strides();
        let mut new_strides = vec![0; input.rank()];
        let mut new_shape = vec![0; input.rank()];

        for (new_axis, &axis) in axes.iter().enumerate() {
            new_shape[new_axis] = shape[axis];
            new_strides[new_axis] = strides[axis];
        }

        ensure!(
            output.shape() == new_shape,
            "Mismatch between expected new shape {:?} and output shape {:?}",
            new_shape,
            output.shape()
        );

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| format!("Unsupported rank {:?} for PermuteAxes", input.rank()))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let out_shape = output.shape();
        let func = cuda_context().load_pipeline(LibraryName::ArrayOps, kernel_name)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&new_strides);
        launch_args.set_slice(out_shape);
        launch_args.set_slice(output.strides());

        let cfg = utils::cuda_launch_cfg_for_cpy(out_shape);

        unsafe { launch_args.launch(cfg) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::context::CUDA_STREAM;

    use super::*;

    use num_traits::Zero;

    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;

    fn run_test_case<F: Datum + Zero + Copy>(shape: &[usize], axes: &[usize]) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_len = shape.iter().product::<usize>();
            let a_data = (0..a_len).map(|f| f as f32).collect::<Vec<_>>();

            let a = Tensor::from_shape(shape, &a_data)?.into_device()?;

            let output = PermuteAxes.eval(stream, &a, axes)?;
            let ref_output = a.to_host()?.into_tensor().permute_axes(axes)?;
            assert_eq!(ref_output, output.to_host()?.into_tensor());
            Ok(())
        })
    }

    #[test]
    fn test_permute_axes() -> TractResult<()> {
        run_test_case::<f32>(&[3, 4], &[1, 0])?;
        run_test_case::<f32>(&[1, 2, 3, 4, 5], &[4, 1, 2, 3, 0])?;
        run_test_case::<f32>(&[1, 1, 2, 2, 3, 2], &[0, 3, 1, 2, 4, 5])?;
        Ok(())
    }
}

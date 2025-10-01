use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view, launch_args};
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

static WARP_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaledMaskedSoftmax;

impl ScaledMaskedSoftmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType, block_size: usize) -> TractResult<String> {
        ensure!(
            Self::is_supported_dt(dt),
            "Unsupported dt {:?} for cuda scaled masked softmaxop",
            dt
        );
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("scaled_masked_softmax_{block_size}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, scale, mask, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(mask.rank() == 3 && input.rank() == 3);
        ensure!(output.datum_type() == input.datum_type());

        let shape = input.shape();
        let strides = input.strides();
        let mask_strides_nd3 = compute_broadcast_strides::<usize>(mask.shape(), mask.strides())?;

        let i_view = get_cuda_view(input);
        let mask_view = get_cuda_view(mask);
        let o_view = get_cuda_view(output);

        let smbpo = cuda_context().properties().sharedMemPerBlockOptin;
        let nbytes_shared = (shape[2].div_ceil(WARP_SIZE) * WARP_SIZE + WARP_SIZE) * size_of::<f32>();
        ensure!(nbytes_shared < smbpo, "Time to implement GGML fallback");

        let mut nth = 32;
        while nth < shape[2] && nth < MAX_THREADS {
            nth *= 2;
        }

        let block_size =
            if shape[2].is_power_of_two() && shape[2] > 32 { shape[2].min(1024) } else { 0 };

        let func = cuda_context()
            .load_pipeline(LibraryName::NN, self.kernel_name(input.datum_type(), block_size)?)?;

        func.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smbpo as i32,
        );
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&mask_view);

        if input.datum_type() == DatumType::F32 {
            launch_args.arg(scale.to_scalar::<f32>()?)
        } else {
            launch_args.arg(scale.to_scalar::<f16>()?)
        };
        launch_args.arg(&o_view);
        launch_args.set_slice(shape);
        launch_args.set_slice(strides);
        launch_args.set_slice(&mask_strides_nd3);
        launch_args.set_slice(output.strides());

        let cfg = LaunchConfig {
            grid_dim: (1, shape[1] as _, shape[0] as _),
            block_dim: (nth as _, 1, 1),
            shared_mem_bytes: nbytes_shared as u32,
        };

        unsafe { launch_args.launch(cfg) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tract_gpu::tensor::IntoDevice;

    use crate::context::CUDA_STREAM;

    use super::*;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::strategy::Strategy;
    use tract_core::internal::Tensor;
    use tract_transformers::ops::scaled_masked_softmax;

    #[test]
    fn test_scaled_masked_softmax_f32() -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let m = 6;
            let n = 33;
            let scale: Arc<_> = tensor0(0.125f32).into();
            let mask = Tensor::from_shape(&[1, m, n], &vec![-1000f32; m * n])?.into_device()?;

            let a = Tensor::from_shape(
                &[4, m, n],
                &(0..4 * m * n).map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cpu = scaled_masked_softmax::ScaledMaskedSoftmax { scale: scale.clone() };

            let cpu_output = cpu
                .eval(tvec![a.to_host()?.into_tvalue(), mask.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let cuda_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    proptest::proptest! {
        #[test]
        fn scaled_masked_softmax_prop_f32(pb in any::<ScaledMaskedSoftmaxProblem<f32>>()) {
            fn run(pb: ScaledMaskedSoftmaxProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn scaled_masked_softmax_prop_f16(pb in any::<ScaledMaskedSoftmaxProblem<f16>>()) {
            fn run(pb: ScaledMaskedSoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }

    #[derive(Debug, new)]
    pub struct ScaledMaskedSoftmaxProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        pub shape: Vec<usize>,
        pub mask_shape: Vec<usize>,
        pub input: Vec<F>,
        pub mask: Vec<F>,
    }

    impl<F> Arbitrary for ScaledMaskedSoftmaxProblem<F>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (1..5usize, 1usize..5, 5u32..15, 0..2usize)
                .prop_map(|(x, y, exp, noise)| {
                    let shape = vec![x, y, 2usize.pow(exp) + noise];
                    let mut mask_shape = shape.clone();
                    mask_shape[0] = 1;

                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();

                    let mask = (0..mask_shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { shape, input, mask_shape, mask }
                })
                .boxed()
        }
    }

    impl<F> ScaledMaskedSoftmaxProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;
            let mask = Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?;
            let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();

            let cpu_output = scaled_masked_softmax::ScaledMaskedSoftmax { scale }
                .eval(tvec![a.into_tvalue(), mask.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let mask =
                    Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?.into_device()?;
                let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();
                let cuda_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

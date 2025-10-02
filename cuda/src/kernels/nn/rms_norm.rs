use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, MAX_THREADS, WARP_SIZE, get_cuda_view, utils};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNorm;

impl RmsNorm {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType, n_cols: usize) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda rmsop", dt);
        let tname = DeviceTensor::tname(dt)?;
        if n_cols < MAX_THREADS {
            Ok(format!("rms_norm_small_{tname}"))
        } else {
            Ok(format!("rms_norm_{tname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        axis: usize,
        eps: &Tensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, axis, eps, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        axis: usize,
        eps: &Tensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        let kernel_name = self.kernel_name(input.datum_type(), shape_nd3[1])?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let func = cuda_context().load_pipeline(LibraryName::NN, kernel_name)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&shape_nd3);
        launch_args.set_slice(&strides_nd3);
        launch_args.arg(eps.to_scalar::<f32>()?);

        let cfg = LaunchConfig {
            grid_dim: ((shape_nd3[2] * shape_nd3[0]) as _, 1, 1),
            block_dim: if shape_nd3[1] < MAX_THREADS {
                (WARP_SIZE as _, 1, 1)
            } else {
                (MAX_THREADS as _, 1, 1)
            },
            shared_mem_bytes: 0,
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
    use tract_core::internal::Tensor;
    use tract_transformers::ops::rms_norm;

    fn test_case<F>(shape: &[usize], axis: usize, offset: f32, scale: f32) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
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

            let eps = Arc::new(tensor0(0.0001f32));
            let cpu_rms = rms_norm::RmsNorm { axis, eps: Arc::clone(&eps) };

            let cpu_output =
                cpu_rms.eval(tvec![a.to_host()?.into_tvalue()])?[0].clone().into_tensor();
            let cuda_output = RmsNorm.eval(stream, &a, axis, &eps)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)
                .with_context(|| {
                    format!(
                        "Input: {:?}, Cpu: {:?}, Cuda: {:?}",
                        a.to_host().and_then(|it| it.dump(true)),
                        cpu_output.dump(true),
                        cuda_output.to_host().and_then(|it| it.dump(true))
                    )
                })?;
            Ok(())
        })
    }

    #[test]
    fn test_rms() -> TractResult<()> {
        test_case::<f32>(&[2, 2], 1, -0.0, 1.0 / 100.0)?;
        test_case::<f16>(&[2, 7], 0, -0.0, 1.0 / 100.0)?;
        test_case::<f32>(&[2, 124], 1, -0.0, 1.0 / 100.0)?;
        test_case::<f16>(&[1026, 7], 0, -0.0, 1.0 / 100.0)?;
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn rms_prop_f32(pb in any::<RmsNormProblem<f32>>()) {
            fn run(pb: RmsNormProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn rms_prop_f16(pb in any::<RmsNormProblem<f16>>()) {
            fn run(pb: RmsNormProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }

    #[derive(Debug, new)]
    pub struct RmsNormProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub shape: Vec<usize>,
        pub axis: usize,
        pub input: Vec<F>,
        pub eps: Arc<Tensor>,
    }

    impl<F> Arbitrary for RmsNormProblem<F>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (0usize..5, 0usize..1)
                .prop_flat_map(|(left, right)| {
                    let axis = left;
                    let shape_len = usize::min(left + right, 4);
                    let iter_ax_dim = 1usize..1024;
                    let other_dim = 1usize..10;
                    (iter_ax_dim, vec(other_dim, shape_len..=shape_len), Just(axis))
                })
                .prop_map(|(iter_dim, mut shape, axis)| {
                    shape.insert(axis, iter_dim);
                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { shape, axis, input, eps: Arc::new(tensor0(0.0001f32)) }
                })
                .boxed()
        }
    }

    impl<F> RmsNormProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_rms = rms_norm::RmsNorm { axis: self.axis, eps: Arc::clone(&self.eps) };

            let cpu_output = cpu_rms.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();

            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let cuda_output = RmsNorm.eval(stream, &a, self.axis, &self.eps)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

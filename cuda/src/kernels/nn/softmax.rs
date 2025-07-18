use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view, launch_args, utils};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Softmax;

impl Softmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType, n_cols: usize) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda softmaxop", dt);
        let tname = DeviceTensor::tname(dt)?;
        if n_cols < MAX_THREADS {
            Ok(format!("softmax_small_{tname}"))
        } else {
            Ok(format!("softmax_{tname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        axis: usize,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, axis, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        axis: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let func = cuda_context()
            .load_pipeline(LibraryName::NN, self.kernel_name(input.datum_type(), shape_nd3[1])?)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&shape_nd3);
        launch_args.set_slice(&strides_nd3);

        let cfg = LaunchConfig {
            grid_dim: ((shape_nd3[0] * shape_nd3[2]) as _, 1, 1),
            block_dim: if shape_nd3[1] < MAX_THREADS {
                (32, 1, 1)
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
    use crate::context::CUDA_STREAM;

    use super::*;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;
    use tract_core::ops::nn::Softmax as TractSoftmax;
    use tract_core::ops::nn::{SoftmaxExp, SoftmaxKind};
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn test_softmax_f32() -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let m = 2;
            let k = 3;
            let axis = 1;

            let a = Tensor::from_shape(&[m, k], &(0..m * k).map(|f| f as f32).collect::<Vec<_>>())?
                .into_device()?;

            let cpu_softmax = TractSoftmax {
                axes: tvec![axis],
                quant_output_dt: None,
                kind: SoftmaxKind::Softmax(SoftmaxExp::Libc),
            };

            let cpu_output =
                cpu_softmax.eval(tvec![a.to_host()?.into_tvalue()])?[0].clone().into_tensor();
            let cuda_output = Softmax.eval(stream, &a, axis)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_f32_2() -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let shape = [8, 4, 3];
            let num_elements = shape.iter().product();
            let axis = 0;

            let a = Tensor::from_shape(
                &shape,
                &(0..num_elements).map(|f| f as f32 / 1000.0).collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cpu_softmax = TractSoftmax {
                axes: tvec![axis],
                quant_output_dt: None,
                kind: SoftmaxKind::Softmax(SoftmaxExp::Libc),
            };

            let cpu_output =
                cpu_softmax.eval(tvec![a.to_host()?.into_tvalue()])?[0].clone().into_tensor();
            let cuda_output = Softmax.eval(stream, &a, axis)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_f16() -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let m = 4;
            let k = 4;
            let axis = 1;

            let a = Tensor::from_shape(
                &[m, k],
                &(0..m * k).map(|f| -> f16 { f.as_() }).collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cpu_softmax = TractSoftmax {
                axes: tvec![axis],
                quant_output_dt: None,
                kind: SoftmaxKind::Softmax(SoftmaxExp::Libc),
            };

            let cpu_output =
                cpu_softmax.eval(tvec![a.to_host()?.into_tvalue()])?[0].clone().into_tensor();
            let cuda_output = Softmax.eval(stream, &a, axis)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    proptest::proptest! {
        #[test]
        fn softmax_prop_f32(pb in any::<SoftmaxProblem<f32>>()) {
            fn run(pb: SoftmaxProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn softmax_prop_f16(pb in any::<SoftmaxProblem<f16>>()) {
            fn run(pb: SoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }

    #[derive(Debug, new)]
    pub struct SoftmaxProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        pub shape: Vec<usize>,
        pub axis: usize,
        pub input: Vec<F>,
    }

    impl<F> Arbitrary for SoftmaxProblem<F>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (0usize..3, 0usize..3)
                .prop_flat_map(|(left, right)| {
                    let axis = left;
                    let shape_len = usize::min(left + right + 1, 4);
                    let shape = 1usize..10;
                    (vec(shape, shape_len..=shape_len), Just(axis))
                })
                .prop_map(|(shape, axis)| {
                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { shape, axis, input }
                })
                .boxed()
        }
    }

    impl<F> SoftmaxProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_softmax = TractSoftmax {
                axes: tvec![self.axis],
                quant_output_dt: None,
                kind: SoftmaxKind::Softmax(SoftmaxExp::Libc),
            };
            let cpu_output = cpu_softmax.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();
            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let cuda_output = Softmax.eval(stream, &a, self.axis)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

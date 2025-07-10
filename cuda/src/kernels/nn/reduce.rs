use crate::context::cuda_context;
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{get_cuda_view, launch_args, utils, LibraryName, MAX_THREADS};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reducer {
    MeanOfSquares,
    Sum,
    Prod,
    Min,
    Max,
}

impl Reducer {
    pub const ALL: [Reducer; 5] =
        [Self::MeanOfSquares, Self::Sum, Self::Prod, Self::Min, Self::Max];

    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType, n_cols: usize) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda reduceop", dt);
        let tname = DeviceTensor::tname(dt)?;
        let op = match self {
            Self::MeanOfSquares => "mean_of_squares",
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Min => "min",
            Self::Max => "max",
        };
        if n_cols < 1024 {
            Ok(format!("reduce_{op}_small_{tname}"))
        } else {
            Ok(format!("reduce_{op}_{tname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        axis: usize,
    ) -> TractResult<DeviceTensor> {
        let mut o_shape = input.shape().to_vec();
        o_shape[axis] = 1;
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), &o_shape)? };
        self.dispatch_eval(stream, input, axis, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        axis: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.datum_type() == input.datum_type());
        ensure!(output.shape()[axis] == 1);

        let input_shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let input_strides_nd3 = Tensor::natural_strides(&input_shape_nd3);
        let output_shape_nd3 = utils::reshape_to_rank_3(output.shape(), axis);
        let output_strides_nd3 = Tensor::natural_strides(&output_shape_nd3);

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        
        let func = cuda_context().load_pipeline(LibraryName::NN, self.kernel_name(input.datum_type(), input_shape_nd3[1])?)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&input_shape_nd3);
        launch_args.set_slice(&input_strides_nd3);
        launch_args.set_slice(&output_strides_nd3);

        let cfg = LaunchConfig {
            grid_dim: (input_shape_nd3[2] as _, 1 as _, input_shape_nd3[0] as _),
            block_dim: if input_shape_nd3[1] < MAX_THREADS { (32 ,1, 1) } else { (MAX_THREADS as _, 1, 1)},
            shared_mem_bytes: 0
        };
        unsafe { launch_args.launch(cfg); }
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
    use tract_core::ops::nn::Reducer as TractReducer;
    use tract_gpu::tensor::IntoDevice;

    fn test_case<F>(
        reducer: Reducer,
        tract_reducer: TractReducer,
        shape: &[usize],
        axis: usize,
        scale: f32,
    ) -> TractResult<()>
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
                        let v: f32 = f.as_() ;
                        (v * scale).as_()
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cpu_output = tract_reducer.reduce(&[axis], &a.to_host()?.into_tensor())?;
            let cuda_output = reducer.eval(stream, &a, axis)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)
                .with_context(|| {
                    format!(
                        "A: {:?}, scale: {:?} Cpu: {:?}, Cuda: {:?}",
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
    fn test_reduce_mean_of_squares() -> TractResult<()> {
        test_case::<f32>(Reducer::MeanOfSquares, TractReducer::MeanOfSquares, &[4, 4], 1, 1.0)?;
        test_case::<f16>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[4, 4],
            1,
            1.0 / 100.0,
        )?;
        test_case::<f16>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[1, 10],
            0,
            1.0 / 100.0,
        )?;
        test_case::<f32>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[1, 10],
            0,
            1.0 / 100.0,
        )?;
        test_case::<f16>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[2, 1],
            1,
            1.0 / 100.0,
        )?;
        test_case::<f32>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[2, 1],
            1,
            1.0 / 100.0,
        )?;
        test_case::<f16>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[2, 2, 82, 38],
            1,
            1.0 / 100.0,
        )?;
        test_case::<f16>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[2, 2, 82, 38],
            2,
            1.0 / 100.0,
        )?;
        test_case::<f32>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[2, 2, 82, 38],
            1,
            1.0 / 100.0,
        )?;
        test_case::<f32>(
            Reducer::MeanOfSquares,
            TractReducer::MeanOfSquares,
            &[2, 2, 82, 38],
            2,
            1.0 / 100.0,
        )?;
        Ok(())
    }

    #[test]
    fn test_reduce_sum() -> TractResult<()> {
        test_case::<f32>(Reducer::Sum, TractReducer::Sum, &[4, 4], 1, 1.0)?;
        test_case::<f16>(Reducer::Sum, TractReducer::Sum, &[4, 4], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Sum, TractReducer::Sum, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Sum, TractReducer::Sum, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Sum, TractReducer::Sum, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Sum, TractReducer::Sum, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Sum, TractReducer::Sum, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Sum, TractReducer::Sum, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Sum, TractReducer::Sum, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Sum, TractReducer::Sum, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_reduce_prod() -> TractResult<()> {
        test_case::<f32>(Reducer::Prod, TractReducer::Prod, &[4, 4], 1, 1.0)?;
        test_case::<f16>(Reducer::Prod, TractReducer::Prod, &[4, 4], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Prod, TractReducer::Prod, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Prod, TractReducer::Prod, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Prod, TractReducer::Prod, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Prod, TractReducer::Prod, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Prod, TractReducer::Prod, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Prod, TractReducer::Prod, &[2, 2, 82, 38], 2, 1.0 / 100000.0)?;
        test_case::<f32>(Reducer::Prod, TractReducer::Prod, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Prod, TractReducer::Prod, &[2, 2, 82, 38], 2, 1.0 / 1000.0)?;
        Ok(())
    }

    #[test]
    fn test_reduce_max() -> TractResult<()> {
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 2], 1, 1.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[4, 4], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[1, 10], 0, -1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[2, 1], 1, -1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 1, -1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 2, -1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_reduce_min() -> TractResult<()> {
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[4, 4], 1, 1.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[4, 4], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[1, 10], 0, -1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 1, -1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 1, -1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn reduce_prop_f32(pb in any::<ReduceProblem<f32>>()) {
            fn run(pb: ReduceProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn reduce_prop_f16(pb in any::<ReduceProblem<f16>>()) {
            fn run(pb: ReduceProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }

    #[derive(Debug, new)]
    pub struct ReduceProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        pub op: Reducer,
        pub shape: Vec<usize>,
        pub axis: usize,
        pub input: Vec<F>,
    }

    impl<F> Arbitrary for ReduceProblem<F>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (0..Reducer::ALL.len(), 0usize..3, 0usize..3)
                .prop_flat_map(|(op_idx, left, right)| {
                    let axis = left;
                    let shape_len = usize::min(left + right + 1, 4);
                    let shape = 1usize..10;
                    let op = Reducer::ALL[op_idx];
                    (Just(op), vec(shape, shape_len..=shape_len), Just(axis))
                })
                .prop_map(|(op, shape, axis)| {
                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { op, shape, axis, input }
                })
                .boxed()
        }
    }

    impl<F> ReduceProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;
            let cpu_output = match self.op {
                Reducer::Sum => TractReducer::Sum.reduce(&[self.axis], &a)?,
                Reducer::Prod => TractReducer::Prod.reduce(&[self.axis], &a)?,
                Reducer::MeanOfSquares => TractReducer::MeanOfSquares.reduce(&[self.axis], &a)?,
                Reducer::Min => TractReducer::Min.reduce(&[self.axis], &a)?,
                Reducer::Max => TractReducer::Max.reduce(&[self.axis], &a)?,
            };
            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let cuda_output = self.op.eval(stream, &a, self.axis)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

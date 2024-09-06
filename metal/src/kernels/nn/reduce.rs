use crate::kernels::utils;
use crate::{LibraryName, MetalContext, MetalTensor};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;

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
        Self::tname(dt).is_ok()
    }

    pub fn tname(dt: DatumType) -> Result<&'static str> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            _ => bail!("Unsupport dt {:?} for reducer op", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        let tname = Self::tname(dt)?;
        let op = match self {
            Self::MeanOfSquares => "mean_of_squares",
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Min => "min",
            Self::Max => "max",
        };
        Ok(format!("nn_ops::reduce_{op}_nd3_{tname}"))
    }

    pub fn reshape_to_rank_3(shape: &[usize], axis: usize) -> Vec<usize> {
        let dim_axis_0 = shape[0..axis].iter().product::<usize>();
        let dim_axis_1 = shape[axis];
        let dim_axis_2 = shape[axis + 1..].iter().product::<usize>();

        vec![dim_axis_0, dim_axis_1, dim_axis_2]
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axis: usize,
    ) -> Result<MetalTensor> {
        let o = self.dispatch_eval(context, input, axis)?;
        context.wait_until_completed()?;
        Ok(o)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axis: usize,
    ) -> Result<MetalTensor> {
        input.retain_until_completion();

        let o_dt = input.datum_type();
        let mut o_shape = input.shape().to_vec();
        o_shape[axis] = 1;

        let input_shape_nd3 = Self::reshape_to_rank_3(input.shape(), axis);
        let input_strides_nd3 = Tensor::natural_strides(&input_shape_nd3);
        let output_shape_nd3 = Self::reshape_to_rank_3(&o_shape, axis);
        let output_strides_nd3 = Tensor::natural_strides(&output_shape_nd3);

        let output = unsafe { MetalTensor::uninitialized_dt(o_dt, &o_shape)? };
        output.retained_until_completion();

        let pipeline = context
            .shared_context()
            .load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;

        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(output.metal()), 0);
        encoder.set_bytes(
            2,
            (input_shape_nd3.len() * std::mem::size_of::<usize>()) as _,
            input_shape_nd3.as_ptr() as *const _,
        );
        encoder.set_bytes(
            3,
            (input_strides_nd3.len() * std::mem::size_of::<usize>()) as _,
            input_strides_nd3.as_ptr() as *const _,
        );
        encoder.set_bytes(
            4,
            (output_strides_nd3.len() * std::mem::size_of::<usize>()) as _,
            output_strides_nd3.as_ptr() as *const _,
        );

        let grid_size = utils::build_metal_size_for_shape(&output_shape_nd3);
        let group_size =
            MTLSize { width: usize::min(32, input_shape_nd3[1]) as _, height: 1, depth: 1 };
        encoder.use_resource(input.metal(), metal::MTLResourceUsage::Read);
        encoder.use_resource(output.metal(), metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;
    use tract_core::ops::nn::Reducer as TractReducer;

    fn test_case<F>(
        reducer: Reducer,
        tract_reducer: TractReducer,
        shape: &[usize],
        axis: usize,
        scale: f32,
    ) -> Result<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let len = shape.iter().product::<usize>();

                let a = Tensor::from_shape(
                    shape,
                    &(0..len)
                        .map(|f| -> F {
                            let v: f32 = f.as_();
                            (v * scale).as_()
                        })
                        .collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let cpu_output = tract_reducer.reduce(&[axis], &a.to_cpu())?;
                let metal_output = reducer.eval(context, &a, axis)?;
                cpu_output
                    .close_enough(&metal_output.to_cpu(), Approximation::Approximate)
                    .with_context(|| {
                        anyhow!(
                            "A: {:?}, scale: {:?} Cpu: {:?}, Metal: {:?}",
                            a.to_cpu().dump(true),
                            scale,
                            cpu_output.dump(true),
                            metal_output.to_cpu().dump(true)
                        )
                    })?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_reduce_mean_of_squares() -> Result<()> {
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
    fn test_reduce_sum() -> Result<()> {
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
    fn test_reduce_prod() -> Result<()> {
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
    fn test_reduce_max() -> Result<()> {
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[4, 4], 1, 1.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[4, 4], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Max, TractReducer::Max, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_reduce_min() -> Result<()> {
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[4, 4], 1, 1.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[4, 4], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[1, 10], 0, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[2, 1], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
        test_case::<f16>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 2, 1.0 / 100.0)?;
        test_case::<f32>(Reducer::Min, TractReducer::Min, &[2, 2, 82, 38], 1, 1.0 / 100.0)?;
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
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn reduce_prop_f16(pb in any::<ReduceProblem<f16>>()) {
            fn run(pb: ReduceProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
        pub fn reference(&self) -> Result<Tensor> {
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

        pub fn run(&self) -> Result<Tensor> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_metal()?;
                    let metal_output = self.op.eval(context, &a, self.axis)?;
                    Ok(metal_output.to_cpu())
                })
            })
        }
    }
}

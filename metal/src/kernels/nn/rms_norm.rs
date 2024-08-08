use crate::{LibraryName, MetalContext, MetalTensor};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNorm;

impl RmsNorm {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal rms  op", dt);
        let tname = MetalTensor::tname(dt)?;
        Ok(format!("nn_ops::rms_norm_nd3_{tname}"))
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
        eps: &Tensor,
    ) -> Result<MetalTensor> {
        let o = self.dispatch_eval(context, input, axis, eps)?;
        context.wait_until_completed()?;
        Ok(o)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axis: usize,
        eps: &Tensor,
    ) -> Result<MetalTensor> {
        input.retained_until_completion();

        let o_dt = input.datum_type();
        let o_shape = input.shape().to_vec();

        let shape_nd3 = Self::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        let output = unsafe { MetalTensor::uninitialized_dt(o_dt, &o_shape)? };
        output.retained_until_completion();

        let pipeline = context
            .shared_context()
            .load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;

        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder
            .set_bytes(1, o_dt.size_of() as _, unsafe { eps.as_ptr_unchecked::<u8>() } as *const _);
        encoder.set_buffer(2, Some(output.metal()), 0);
        encoder.set_bytes(
            3,
            (shape_nd3.len() * std::mem::size_of::<usize>()) as _,
            shape_nd3.as_ptr() as *const _,
        );
        encoder.set_bytes(
            4,
            (strides_nd3.len() * std::mem::size_of::<usize>()) as _,
            strides_nd3.as_ptr() as *const _,
        );

        let grid_size = MTLSize { width: shape_nd3[2] as _, height: 1, depth: shape_nd3[0] as _ };
        let group_size = MTLSize { width: usize::min(32, shape_nd3[1]) as _, height: 1, depth: 1 };

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
    use crate::rewrite_rules::BasicRmsNorm;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;

    fn test_case<F>(shape: &[usize], axis: usize, offset: f32, scale: f32) -> Result<()>
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
                            (v * scale + offset).as_()
                        })
                        .collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let eps = Arc::new(tensor0(0.0001f32.as_()));
                let cpu_rms = BasicRmsNorm { axis, eps: Arc::clone(&eps) };

                let cpu_output =
                    cpu_rms.eval(tvec![a.to_cpu().into_tvalue()])?[0].clone().into_tensor();
                let metal_output = RmsNorm.eval(context, &a, axis, &eps)?;

                cpu_output
                    .close_enough(&metal_output.to_cpu(), Approximation::Approximate)
                    .with_context(|| {
                        anyhow!(
                            "Input: {:?}, scale: {:?} Cpu: {:?}, Metal: {:?}",
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
    fn test_rms() -> Result<()> {
        test_case::<f32>(&[4, 4], 1, -8.0, 1.0 / 100.0)?;
        test_case::<f16>(&[4, 4], 1, -8.0, 1.0 / 100.0)?;
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn rms_prop_f32(pb in any::<RmsNormProblem<f32>>()) {
            fn run(pb: RmsNormProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn rms_prop_f16(pb in any::<RmsNormProblem<f16>>()) {
            fn run(pb: RmsNormProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
                    Self { shape, axis, input, eps: Arc::new(tensor0(0.0001f32.as_())) }
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
        pub fn reference(&self) -> Result<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_rms = BasicRmsNorm { axis: self.axis, eps: Arc::clone(&self.eps) };

            let cpu_output = cpu_rms.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();

            Ok(cpu_output)
        }

        pub fn run(&self) -> Result<Tensor> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_metal()?;
                    let metal_output = RmsNorm.eval(context, &a, self.axis, &self.eps)?;
                    Ok(metal_output.to_cpu())
                })
            })
        }
    }
}

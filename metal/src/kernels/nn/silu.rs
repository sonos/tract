use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalContext, MetalTensor};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Silu;

impl Silu {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal silu  op", dt);
        let tname = MetalTensor::tname(dt)?;
        Ok(format!("nn_ops::silu_{tname}"))
    }

    pub fn eval(&self, context: &MetalContext, input: &MetalTensor) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(context, input, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<()> {
        input.retain_until_completion();
        output.retained_until_completion();

        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let kernel_name = self.kernel_name(input.datum_type())?;

        let pipeline = context.shared_context().load_pipeline(LibraryName::NNOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);

            let grid_size = MTLSize { width: output.len() as _, height: 1, depth: 1 };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
            encoder.end_encoding();
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite_rules::BasicSilu;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;

    fn test_case<F>(
        shape: &[usize],
        offset: f32,
        scale: f32,
        appriximate: Approximation,
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
                            (v * scale + offset).as_()
                        })
                        .collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let cpu_output =
                    BasicSilu.eval(tvec![a.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
                let metal_output = Silu.eval(context, &a)?;

                cpu_output.close_enough(&metal_output.to_cpu()?, appriximate).with_context(
                    || {
                        anyhow!(
                            "Input: {:?}, scale: {:?} Cpu: {:?}, Metal: {:?}",
                            a.to_cpu().and_then(|it| it.dump(true)),
                            scale,
                            cpu_output.dump(true),
                            metal_output.to_cpu().and_then(|it| it.dump(true))
                        )
                    },
                )?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_silu() -> Result<()> {
        test_case::<f32>(&[4, 4], -0.0, 1.0 / 100.0, Approximation::Approximate)?;
        test_case::<f16>(&[4, 4], -6.0, 1.0 / 1000.0, Approximation::SuperApproximate)?;
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn silu_prop_f32(pb in any::<SiluProblem<f32>>()) {
            fn run(pb: SiluProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn silu_prop_f16(pb in any::<SiluProblem<f16>>()) {
            fn run(pb: SiluProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }

    #[derive(Debug, new)]
    pub struct SiluProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub shape: Vec<usize>,
        pub input: Vec<F>,
    }

    impl<F> Arbitrary for SiluProblem<F>
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
                    let shape_len = usize::min(left + right + 1, 4);
                    let shape = 1usize..10;
                    vec(shape, shape_len..=shape_len)
                })
                .prop_map(|shape| {
                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { shape, input }
                })
                .boxed()
        }
    }

    impl<F> SiluProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> Result<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_output = BasicSilu.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();

            Ok(cpu_output)
        }

        pub fn run(&self) -> Result<Tensor> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_metal()?;
                    let metal_output = Silu.eval(context, &a)?;
                    metal_output.to_cpu()
                })
            })
        }
    }
}

use crate::encoder::EncoderExt;
use crate::kernels::utils;
use crate::{LibraryName, MetalContext, MetalTensor};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Softmax;

impl Softmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal softmax  op", dt);
        let tname = MetalTensor::tname(dt)?;
        Ok(format!("nn_ops::softmax_nd3_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axis: usize,
    ) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(context, input, axis, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axis: usize,
        output: &MetalTensor,
    ) -> Result<()> {
        input.retained_until_completion();
        output.retained_until_completion();

        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        let pipeline = context
            .shared_context()
            .load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;

        let command_buffer = context.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(2, &shape_nd3);
            encoder.set_slice(3, &strides_nd3);

            let grid_size =
                MTLSize { width: shape_nd3[2] as _, height: 1, depth: shape_nd3[0] as _ };
            let group_size =
                MTLSize { width: usize::min(32, shape_nd3[1]) as _, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
            encoder.end_encoding();
        });
        Ok(())
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
    use tract_core::ops::nn::Softmax as TractSoftmax;
    use tract_core::ops::nn::SoftmaxExp;

    #[test]
    fn test_softmax_f32() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let m = 4;
                let k = 4;
                let axis = 1;

                let a =
                    Tensor::from_shape(&[m, k], &(0..m * k).map(|f| f as f32).collect::<Vec<_>>())?
                        .into_metal()?;

                let cpu_softmax = TractSoftmax {
                    axes: tvec![axis],
                    quant_output_dt: None,
                    exp: SoftmaxExp::Libc,
                };

                let cpu_output =
                    cpu_softmax.eval(tvec![a.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
                let metal_output = Softmax.eval(context, &a, axis)?;
                cpu_output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_softmax_f32_2() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let shape = [8, 4, 3];
                let num_elements = shape.iter().product();
                let axis = 0;

                let a = Tensor::from_shape(
                    &shape,
                    &(0..num_elements).map(|f| f as f32 / 1000.0).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let cpu_softmax = TractSoftmax {
                    axes: tvec![axis],
                    quant_output_dt: None,
                    exp: SoftmaxExp::Libc,
                };

                let cpu_output =
                    cpu_softmax.eval(tvec![a.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
                let metal_output = Softmax.eval(context, &a, axis)?;
                cpu_output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_softmax_f16() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let m = 4;
                let k = 4;
                let axis = 1;

                let a = Tensor::from_shape(
                    &[m, k],
                    &(0..m * k).map(|f| -> f16 { f.as_() }).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let cpu_softmax = TractSoftmax {
                    axes: tvec![axis],
                    quant_output_dt: None,
                    exp: SoftmaxExp::Libc,
                };

                let cpu_output =
                    cpu_softmax.eval(tvec![a.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
                let metal_output = Softmax.eval(context, &a, axis)?;
                cpu_output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
                Ok(())
            })
        })
    }

    proptest::proptest! {
        #[test]
        fn softmax_prop_f32(pb in any::<SoftmaxProblem<f32>>()) {
            fn run(pb: SoftmaxProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn softmax_prop_f16(pb in any::<SoftmaxProblem<f16>>()) {
            fn run(pb: SoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
        pub fn reference(&self) -> Result<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_softmax = TractSoftmax {
                axes: tvec![self.axis],
                quant_output_dt: None,
                exp: SoftmaxExp::Libc,
            };
            let cpu_output = cpu_softmax.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();
            Ok(cpu_output)
        }

        pub fn run(&self) -> Result<Tensor> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_metal()?;
                    let metal_output = Softmax.eval(context, &a, self.axis)?;
                    metal_output.to_cpu()
                })
            })
        }
    }
}

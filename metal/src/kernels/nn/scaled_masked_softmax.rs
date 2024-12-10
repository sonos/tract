use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalContext, MetalTensor};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaledMaskedSoftmax;

impl ScaledMaskedSoftmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(
            Self::is_supported_dt(dt),
            "Unsupport dt {:?} for metal scaled masked softmax  op",
            dt
        );
        let tname = MetalTensor::tname(dt)?;
        Ok(format!("nn_ops::scaled_masked_softmax_nd3_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        scale: &Tensor,
        mask: &MetalTensor,
    ) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        dbg!(&output);
        self.dispatch_eval(context, input, scale, mask, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        scale: &Tensor,
        mask: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<()> {
        input.retained_until_completion();
        mask.retained_until_completion();
        output.retained_until_completion();

        ensure!(output.shape() == input.shape());
        ensure!(mask.rank() == 3 && input.rank() == 3);
        ensure!(output.datum_type() == input.datum_type());

        let shape = input.shape();
        let strides = input.strides();
        let mask_strides_nd3 =
            crate::utils::compute_broadcast_strides::<usize>(mask.shape(), mask.strides())?;
        
        let pipeline = context
            .shared_context()
            .load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;

        let command_buffer = context.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, mask, metal::MTLResourceUsage::Read);
            encoder.set_tensor(2, scale);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, &shape);
            encoder.set_slice(5, &strides);
            encoder.set_slice(6, &mask_strides_nd3);

            let grid_size = MTLSize { width: 1 as _, height: shape[1] as _, depth: shape[0] as _};
            let group_size = MTLSize { width: usize::min(32, shape[2]) as _, height: 1, depth: 1 };

            encoder.dispatch_thread_groups(grid_size, group_size);
            encoder.end_encoding();
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::rewrite_rules::BasicScaledMaskedSoftmax;
use super::*;
    use crate::IntoMetal;
    use derive_new::new;
    
    
    
    
    use tract_core::internal::Tensor;

    #[test]
    fn test_scaled_masked_softmax_f32() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let m = 4;
                let n = 4;
                let scale: Arc<_> = tensor0(0.125f32).into();
                let mask = Tensor::from_shape(&[1, m, n], &vec![-1000f32; m*n])?.into_metal()?;

                let a =
                    Tensor::from_shape(&[1, m, n], &(0..m * n).map(|f| f as f32).collect::<Vec<_>>())?
                        .into_metal()?;

                let cpu = BasicScaledMaskedSoftmax { scale:scale.clone() };

                let cpu_output =
                    cpu.eval(tvec![a.to_cpu()?.into_tvalue(), mask.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
                let metal_output = ScaledMaskedSoftmax.eval(context, &a, &scale, &mask)?;
                cpu_output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
                Ok(())
            })
        })
    }
}

//     #[test]
//     fn test_softmax_f32_2() -> Result<()> {
//         objc::rc::autoreleasepool(|| {
//             crate::METAL_CONTEXT.with_borrow(|context| {
//                 let shape = [8, 4, 3];
//                 let num_elements = shape.iter().product();
//                 let axis = 0;

//                 let a = Tensor::from_shape(
//                     &shape,
//                     &(0..num_elements).map(|f| f as f32 / 1000.0).collect::<Vec<_>>(),
//                 )?
//                 .into_metal()?;

//                 let cpu_softmax = TractSoftmax {
//                     axes: tvec![axis],
//                     quant_output_dt: None,
//                     exp: SoftmaxExp::Libc,
//                 };

//                 let cpu_output =
//                     cpu_softmax.eval(tvec![a.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
//                 let metal_output = Softmax.eval(context, &a, axis)?;
//                 cpu_output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
//                 Ok(())
//             })
//         })
//     }

//     #[test]
//     fn test_softmax_f16() -> Result<()> {
//         objc::rc::autoreleasepool(|| {
//             crate::METAL_CONTEXT.with_borrow(|context| {
//                 let m = 4;
//                 let k = 4;
//                 let axis = 1;

//                 let a = Tensor::from_shape(
//                     &[m, k],
//                     &(0..m * k).map(|f| -> f16 { f.as_() }).collect::<Vec<_>>(),
//                 )?
//                 .into_metal()?;

//                 let cpu_softmax = TractSoftmax {
//                     axes: tvec![axis],
//                     quant_output_dt: None,
//                     exp: SoftmaxExp::Libc,
//                 };

//                 let cpu_output =
//                     cpu_softmax.eval(tvec![a.to_cpu()?.into_tvalue()])?[0].clone().into_tensor();
//                 let metal_output = Softmax.eval(context, &a, axis)?;
//                 cpu_output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
//                 Ok(())
//             })
//         })
//     }

//     proptest::proptest! {
//         #[test]
//         fn softmax_prop_f32(pb in any::<SoftmaxProblem<f32>>()) {
//             fn run(pb: SoftmaxProblem<f32>) -> TractResult<()> {
//                 let out = pb.run()?;
//                 let reference = pb.reference()?;

//                 out.close_enough(&reference, Approximation::Approximate)
//                    .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
//             }
//             run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
//         }

//         #[test]
//         fn softmax_prop_f16(pb in any::<SoftmaxProblem<f16>>()) {
//             fn run(pb: SoftmaxProblem<f16>) -> TractResult<()> {
//                 let out = pb.run()?;
//                 let reference = pb.reference()?;

//                 out.close_enough(&reference, Approximation::Approximate)
//                    .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
//             }

//             run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
//         }
//     }

//     #[derive(Debug, new)]
//     pub struct SoftmaxProblem<F: Datum + Float>
//     where
//         F: Datum + Float,
//         usize: AsPrimitive<F>,
//     {
//         pub shape: Vec<usize>,
//         pub axis: usize,
//         pub input: Vec<F>,
//     }

//     impl<F> Arbitrary for SoftmaxProblem<F>
//     where
//         F: Datum + Float,
//         usize: AsPrimitive<F>,
//     {
//         type Parameters = ();
//         type Strategy = BoxedStrategy<Self>;

//         fn arbitrary_with(_: ()) -> Self::Strategy {
//             (0usize..3, 0usize..3)
//                 .prop_flat_map(|(left, right)| {
//                     let axis = left;
//                     let shape_len = usize::min(left + right + 1, 4);
//                     let shape = 1usize..10;
//                     (vec(shape, shape_len..=shape_len), Just(axis))
//                 })
//                 .prop_map(|(shape, axis)| {
//                     let input = (0..shape.iter().product::<usize>())
//                         .map(|f| f.as_() / 1000.as_())
//                         .collect::<Vec<_>>();
//                     Self { shape, axis, input }
//                 })
//                 .boxed()
//         }
//     }

//     impl<F> SoftmaxProblem<F>
//     where
//         F: Datum + Float + std::ops::AddAssign,
//         usize: AsPrimitive<F>,
//     {
//         pub fn reference(&self) -> Result<Tensor> {
//             let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

//             let cpu_softmax = TractSoftmax {
//                 axes: tvec![self.axis],
//                 quant_output_dt: None,
//                 exp: SoftmaxExp::Libc,
//             };
//             let cpu_output = cpu_softmax.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();
//             Ok(cpu_output)
//         }

//         pub fn run(&self) -> Result<Tensor> {
//             objc::rc::autoreleasepool(|| {
//                 crate::METAL_CONTEXT.with_borrow(|context| {
//                     let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_metal()?;
//                     let metal_output = Softmax.eval(context, &a, self.axis)?;
//                     metal_output.to_cpu()
//                 })
//             })
//         }
//     }
// }

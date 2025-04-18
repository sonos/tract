use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalContext};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("nn_ops::scaled_masked_softmax_nd3_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
    ) -> Result<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(context, input, scale, mask, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
        output: &DeviceTensor,
    ) -> Result<()> {
        context.retain_tensor(input);
        context.retain_tensor(mask);
        context.retain_tensor(output);

        ensure!(output.shape() == input.shape());
        ensure!(mask.rank() == 3 && input.rank() == 3);
        ensure!(output.datum_type() == input.datum_type());

        let shape = input.shape();
        let strides = input.strides();
        let mask_strides_nd3 =
            tract_gpu::utils::compute_broadcast_strides::<usize>(mask.shape(), mask.strides())?;

        let pipeline =
            context.load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;

        let command_buffer = context.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, mask, metal::MTLResourceUsage::Read);
            encoder.set_tensor(2, scale);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, shape);
            encoder.set_slice(5, strides);
            encoder.set_slice(6, &mask_strides_nd3);
            encoder.set_slice(7, output.strides());
            let grid_size = MTLSize { width: 1 as _, height: shape[1] as _, depth: shape[0] as _ };
            let group_size = MTLSize { width: usize::min(32, shape[2]) as _, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::autorelease_pool_init;
    use crate::context::MetalDevice;
    use tract_gpu::tensor::IntoGpu;

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
    fn test_scaled_masked_softmax_f32() -> Result<()> {
        MetalDevice::register()?;
        let _ = autorelease_pool_init();
        crate::METAL_CONTEXT.with_borrow(|context| {
            let m = 4;
            let n = 4;
            let scale: Arc<_> = tensor0(0.125f32).into();
            let mask = Tensor::from_shape(&[1, m, n], &vec![-1000f32; m * n])?.into_gpu()?;

            let a =
                Tensor::from_shape(&[1, m, n], &(0..m * n).map(|f| f as f32).collect::<Vec<_>>())?
                    .into_gpu()?;

            let cpu = scaled_masked_softmax::ScaledMaskedSoftmax { scale: scale.clone() };

            let cpu_output = cpu
                .eval(tvec![a.to_cpu()?.into_tvalue(), mask.to_cpu()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = ScaledMaskedSoftmax.eval(context, &a, &scale, &mask)?;
            cpu_output
                .close_enough(&metal_output.to_cpu()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_scaled_masked_softmax_f32_2() -> Result<()> {
        MetalDevice::register()?;
        let _ = autorelease_pool_init();
        crate::METAL_CONTEXT.with_borrow(|context| {
            let m = 4;
            let n = 1024;
            let scale: Arc<_> = tensor0(0.125f32).into();
            let mask = Tensor::from_shape(&[1, m, n], &vec![-1000f32; m * n])?.into_gpu()?;

            let a =
                Tensor::from_shape(&[1, m, n], &(0..m * n).map(|f| f as f32).collect::<Vec<_>>())?
                    .into_gpu()?;

            let cpu = scaled_masked_softmax::ScaledMaskedSoftmax { scale: scale.clone() };

            let cpu_output = cpu
                .eval(tvec![a.to_cpu()?.into_tvalue(), mask.to_cpu()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = ScaledMaskedSoftmax.eval(context, &a, &scale, &mask)?;
            cpu_output
                .close_enough(&metal_output.to_cpu()?.into_tensor(), Approximation::Approximate)?;
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
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn scaled_masked_softmax_prop_f16(pb in any::<ScaledMaskedSoftmaxProblem<f16>>()) {
            fn run(pb: ScaledMaskedSoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| anyhow!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
            vec(1usize..10, 3..=3)
                .prop_map(|shape| {
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
        pub fn reference(&self) -> Result<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;
            let mask = Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?;
            let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();

            let cpu_output = scaled_masked_softmax::ScaledMaskedSoftmax { scale }
                .eval(tvec![a.into_tvalue(), mask.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            Ok(cpu_output)
        }

        pub fn run(&self) -> Result<Tensor> {
            MetalDevice::register()?;
            let _ = autorelease_pool_init();
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_gpu()?;
                let mask =
                    Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?.into_gpu()?;
                let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();
                let metal_output = ScaledMaskedSoftmax.eval(context, &a, &scale, &mask)?;
                Ok(metal_output.to_cpu()?.into_tensor())
            })
        }
    }
}

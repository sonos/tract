use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Default, Copy, PartialEq, Eq, Hash)]
pub struct GeluApproximate {
    pub fast_impl: bool,
}

impl GeluApproximate {
    pub fn fast() -> Self {
        Self { fast_impl: true }
    }

    pub fn accurate() -> Self {
        Self { fast_impl: false }
    }

    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal gelu  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        if self.fast_impl {
            Ok(format!("nn_ops::gelu_approx_fast_{tname}"))
        } else {
            Ok(format!("nn_ops::gelu_approx_{tname}"))
        }
    }

    pub fn eval(&self, stream: &MetalStream, input: &DeviceTensor) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let kernel_name = self.kernel_name(input.datum_type())?;

        let pipeline = stream.load_pipeline(LibraryName::NNOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            let grid_size = MTLSize { width: output.len() as _, height: 1, depth: 1 };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::gelu_approximate;

    fn test_case<F>(
        gelu_approx: GeluApproximate,
        shape: &[usize],
        offset: f32,
        scale: f32,
        appriximate: Approximation,
    ) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        with_borrowed_metal_stream(|stream| {
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

            let cpu_output = gelu_approximate::GeluApproximate::default()
                .eval(tvec![a.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = gelu_approx.eval(stream, &a)?;

            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), appriximate)
                .with_context(|| {
                    format!(
                        "Input: {:?}, scale: {:?} Cpu: {:?}, Metal: {:?}",
                        a.to_host().and_then(|it| it.dump(true)),
                        scale,
                        cpu_output.dump(true),
                        metal_output.to_host().and_then(|it| it.dump(true))
                    )
                })?;
            Ok(())
        })
    }

    #[test]
    fn test_gelu_approx() -> TractResult<()> {
        test_case::<f32>(
            GeluApproximate::accurate(),
            &[4, 4],
            -0.0,
            1.0 / 100.0,
            Approximation::Approximate,
        )?;
        test_case::<f32>(
            GeluApproximate::accurate(),
            &[4, 4],
            -6.0,
            1.0 / 1000.0,
            Approximation::Approximate,
        )?;
        test_case::<f16>(
            GeluApproximate::accurate(),
            &[4, 4],
            -6.0,
            1.0 / 1000.0,
            Approximation::SuperApproximate,
        )?;
        Ok(())
    }
    #[test]
    fn test_gelu_approx_fast() -> TractResult<()> {
        test_case::<f32>(
            GeluApproximate::fast(),
            &[4, 4],
            -0.0,
            1.0 / 100.0,
            Approximation::SuperApproximate,
        )?;
        test_case::<f32>(
            GeluApproximate::fast(),
            &[4, 4],
            -6.0,
            1.0 / 1000.0,
            Approximation::SuperApproximate,
        )?;
        test_case::<f16>(
            GeluApproximate::fast(),
            &[4, 4],
            -6.0,
            1.0 / 1000.0,
            Approximation::SuperApproximate,
        )?;
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn gelu_approx_prop_f32(pb in any::<GeluProblem<f32>>()) {
            fn run(pb: GeluProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn gelu_approx_prop_f16(pb in any::<GeluProblem<f16>>()) {
            fn run(pb: GeluProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }

    #[derive(Debug, new)]
    pub struct GeluProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub shape: Vec<usize>,
        pub input: Vec<F>,
    }

    impl<F> Arbitrary for GeluProblem<F>
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

    impl<F> GeluProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_output = gelu_approximate::GeluApproximate::default()
                .eval(tvec![a.into_tvalue()])?[0]
                .clone()
                .into_tensor();

            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            with_borrowed_metal_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let metal_output = GeluApproximate::accurate().eval(stream, &a)?;
                Ok(metal_output.to_host()?.into_tensor())
            })
        }
    }
}

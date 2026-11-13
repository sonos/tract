use crate::encoder::EncoderExt;
use crate::kernels::utils;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::nn::SoftmaxKind;
use tract_gpu::tensor::DeviceTensor;

/// Most threads sharing a row of the last axis. A row along an inner axis keeps one simdgroup:
/// its loads do not coalesce, so more threads cost occupancy without speeding it up.
const LAST_AXIS_ROW_THREADS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Softmax;

impl Softmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(
        &self,
        dt: DatumType,
        kind: SoftmaxKind,
        contig: bool,
    ) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal softmaxop", dt);
        let tname = DeviceTensor::tname(dt)?;
        let op = match kind {
            SoftmaxKind::Softmax => "softmax",
            SoftmaxKind::LogSoftmax => "log_softmax",
        };
        let contig = if contig { "_contig" } else { "" };
        Ok(format!("nn_ops::{op}_nd3{contig}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axis: usize,
        kind: SoftmaxKind,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, axis, kind, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axis: usize,
        kind: SoftmaxKind,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        // Fast path for contiguous last-axis softmax (strides[1]==1) whose dim is
        // a multiple of 4 (so every row start stays 16-byte aligned for the
        // float4/half4 loads). Falls back to the general strided kernel otherwise.
        let use_contig = strides_nd3[1] == 1 && shape_nd3[1].is_multiple_of(4);
        let kernel_name = self.kernel_name(input.datum_type(), kind, use_contig)?;
        let pipeline = stream.load_pipeline(LibraryName::NNOps, &kernel_name)?;

        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(2, &shape_nd3);
            encoder.set_slice(3, &strides_nd3);

            let grid_size =
                MTLSize { width: shape_nd3[2] as _, height: 1, depth: shape_nd3[0] as _ };
            let work = if use_contig { shape_nd3[1] / 4 } else { shape_nd3[1] };
            let cap = if shape_nd3[2] == 1 { LAST_AXIS_ROW_THREADS } else { 32 };
            let max = pipeline.max_total_threads_per_threadgroup() as usize;
            let threads =
                if work <= 32 { work.max(1) } else { work.next_multiple_of(32).min(cap).min(max) };
            let group_size = MTLSize { width: threads as _, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_softmax_dispatch(
    input: &DeviceTensor,
    axis: usize,
    kind: SoftmaxKind,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| Softmax.dispatch_eval(stream, input, axis, kind, output))
}

crate::register_metal_op!(tract_core::ops::nn::Softmax, |source, node, op| {
    rule_if!(Softmax::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::softmax::GpuSoftmax::from_tract_core(
        op,
        "Metal",
        metal_softmax_dispatch,
    )?)))
});

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
    use tract_core::ops::nn::Softmax as TractSoftmax;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn test_softmax_f32() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let m = 4;
            let k = 4;
            let axis = 1;

            let a = Tensor::from_shape(&[m, k], &(0..m * k).map(|f| f as f32).collect::<Vec<_>>())?
                .into_device()?;

            let cpu_softmax = TractSoftmax {
                axes: tvec![axis],
                quant_output_dt: None,
                kind: SoftmaxKind::Softmax,
            };

            let cpu_output = cpu_softmax
                .eval(&EvalContext::out_of_plan(), tvec![a.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = Softmax.eval(stream, &a, axis, SoftmaxKind::Softmax)?;
            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_f32_2() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
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
                kind: SoftmaxKind::Softmax,
            };

            let cpu_output = cpu_softmax
                .eval(&EvalContext::out_of_plan(), tvec![a.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = Softmax.eval(stream, &a, axis, SoftmaxKind::Softmax)?;
            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_f16() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
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
                kind: SoftmaxKind::Softmax,
            };

            let cpu_output = cpu_softmax
                .eval(&EvalContext::out_of_plan(), tvec![a.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = Softmax.eval(stream, &a, axis, SoftmaxKind::Softmax)?;
            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_masked_long_rows_f32() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            for kind in [SoftmaxKind::Softmax, SoftmaxKind::LogSoftmax] {
                for k in [7, 300, 4095, 5000, 70000] {
                    let mut input =
                        (0..3 * k).map(|f| ((f * 37) % 101) as f32 / 10.0).collect::<Vec<_>>();
                    for r in 0..3 {
                        input[r * k + (r * 5) % k] = f32::NEG_INFINITY;
                        input[r * k] = f32::NEG_INFINITY;
                    }
                    for (shape, axis) in [([3, k, 1], 1), ([1, k, 3], 1)] {
                        let a = Tensor::from_shape(&shape, &input)?.into_device()?;
                        let cpu = TractSoftmax { axes: tvec![axis], quant_output_dt: None, kind }
                            .eval(
                            &EvalContext::out_of_plan(),
                            tvec![a.to_host()?.into_tvalue()],
                        )?[0]
                            .clone()
                            .into_tensor();
                        let metal = Softmax.eval(stream, &a, axis, kind)?;
                        cpu.close_enough(
                            &metal.to_host()?.into_tensor(),
                            Approximation::Approximate,
                        )
                        .with_context(|| format!("{kind:?} {shape:?}"))?;
                    }
                }
            }
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
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn softmax_prop_f16(pb in any::<SoftmaxProblem<f16>>()) {
            fn run(pb: SoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
        pub kind: SoftmaxKind,
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
                    let kind =
                        prop_oneof![Just(SoftmaxKind::Softmax), Just(SoftmaxKind::LogSoftmax)];
                    (vec(shape, shape_len..=shape_len), Just(axis), kind)
                })
                .prop_map(|(shape, axis, kind)| {
                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { shape, axis, kind, input }
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

            let cpu_softmax =
                TractSoftmax { axes: tvec![self.axis], quant_output_dt: None, kind: self.kind };
            let cpu_output = cpu_softmax
                .eval(&EvalContext::out_of_plan(), tvec![a.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            with_borrowed_metal_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let metal_output = Softmax.eval(stream, &a, self.axis, self.kind)?;
                Ok(metal_output.to_host()?.into_tensor())
            })
        }
    }
}

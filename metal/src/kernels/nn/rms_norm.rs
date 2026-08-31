use crate::encoder::EncoderExt;
use crate::kernels::utils;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RmsNorm;

impl RmsNorm {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(
        &self,
        in_dt: DatumType,
        out_dt: DatumType,
        is_l4: bool,
        scaled: bool,
        with_residual: bool,
    ) -> TractResult<String> {
        ensure!(Self::is_supported_dt(in_dt), "Unsupported dt {:?} for metal rmsop", in_dt);
        ensure!(Self::is_supported_dt(out_dt), "Unsupported out dt {:?} for metal rmsop", out_dt);
        let iname = DeviceTensor::tname(in_dt)?;
        let oname = DeviceTensor::tname(out_dt)?;
        let variant = match (scaled, with_residual) {
            (true, true) => "rms_norm_scaled_add",
            (true, false) => "rms_norm_scaled",
            (false, true) => "rms_norm_add",
            (false, false) => "rms_norm",
        };
        if !is_l4 {
            Ok(format!("nn_ops::{variant}_nd3_{iname}_{oname}"))
        } else {
            Ok(format!("nn_ops::{variant}_nd2_l4_{iname}_{oname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        scale: Option<&DeviceTensor>,
        out_dt: Option<DatumType>,
        axis: usize,
        eps: &Tensor,
    ) -> TractResult<DeviceTensor> {
        let out_dt = out_dt.unwrap_or(input.datum_type());
        let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, input.shape())? };
        self.dispatch_eval(stream, input, None, scale, axis, eps, &output, None)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    /// `residual`/`sum_out` come and go together: the normalized value is
    /// then `input + residual` (computed in the input dtype, bit-identical
    /// to the standalone Add dispatch) and the sum is also written to
    /// `sum_out` for the consumers of the residual stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        residual: Option<&DeviceTensor>,
        scale: Option<&DeviceTensor>,
        axis: usize,
        eps: &Tensor,
        output: &DeviceTensor,
        sum_out: Option<&DeviceTensor>,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);
        if let Some(scale) = scale {
            stream.retain_tensor(scale);
            ensure!(scale.datum_type() == DatumType::F32);
            ensure!(scale.len() == input.shape()[axis]);
        }
        ensure!(residual.is_some() == sum_out.is_some());
        if let (Some(residual), Some(sum_out)) = (residual, sum_out) {
            stream.retain_tensor(residual);
            stream.retain_tensor(sum_out);
            ensure!(residual.shape() == input.shape());
            ensure!(residual.datum_type() == input.datum_type());
            ensure!(sum_out.shape() == input.shape());
            ensure!(sum_out.datum_type() == input.datum_type());
        }

        ensure!(output.shape() == input.shape());
        ensure!(Self::is_supported_dt(output.datum_type()));

        if (axis == (input.rank() - 1)) && input.shape()[axis].is_multiple_of(4) {
            let shape = input.shape();
            let shape_nd2 = tvec![shape[..axis].iter().product::<usize>(), shape[axis]];

            let pipeline = stream.load_pipeline(
                LibraryName::NNOps,
                &self.kernel_name(
                    input.datum_type(),
                    output.datum_type(),
                    true,
                    scale.is_some(),
                    residual.is_some(),
                )?,
            )?;

            let iter_dim = shape_nd2[1];
            let iter_dim_div_4 = iter_dim / 4;
            let outer_stride = iter_dim * input.datum_type().size_of();

            let mut nthreads = 32;
            let limit = iter_dim_div_4.min(pipeline.max_total_threads_per_threadgroup() as usize);

            while (nthreads * 2) < limit {
                nthreads *= 2;
            }
            nthreads = nthreads.min(iter_dim_div_4);
            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                let mut idx = 0;
                encoder.set_metal_tensor(idx, input, metal::MTLResourceUsage::Read);
                idx += 1;
                if let Some(residual) = residual {
                    encoder.set_metal_tensor(idx, residual, metal::MTLResourceUsage::Read);
                    idx += 1;
                }
                encoder.set_tensor(idx, eps);
                idx += 1;
                if let Some(scale) = scale {
                    encoder.set_metal_tensor(idx, scale, metal::MTLResourceUsage::Read);
                    idx += 1;
                }
                encoder.set_metal_tensor(idx, output, metal::MTLResourceUsage::Write);
                idx += 1;
                if let Some(sum_out) = sum_out {
                    encoder.set_metal_tensor(idx, sum_out, metal::MTLResourceUsage::Write);
                    idx += 1;
                }
                encoder.set_bytes(
                    idx,
                    std::mem::size_of::<usize>() as u64,
                    &iter_dim as *const usize as *const _,
                );
                idx += 1;
                encoder.set_bytes(
                    idx,
                    std::mem::size_of::<usize>() as u64,
                    &iter_dim_div_4 as *const usize as *const _,
                );
                idx += 1;
                encoder.set_bytes(
                    idx,
                    std::mem::size_of::<usize>() as u64,
                    &outer_stride as *const usize as *const _,
                );
                encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
                let grid_size = MTLSize { width: shape_nd2[0] as _, height: 1, depth: 1 };
                let group_size = MTLSize { width: nthreads as _, height: 1, depth: 1 };

                encoder.dispatch_thread_groups(grid_size, group_size);
            });
        } else {
            let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
            let strides_nd3 = Tensor::natural_strides(&shape_nd3);

            let pipeline = stream.load_pipeline(
                LibraryName::NNOps,
                &self.kernel_name(
                    input.datum_type(),
                    output.datum_type(),
                    false,
                    scale.is_some(),
                    residual.is_some(),
                )?,
            )?;

            let iter_dim = shape_nd3[1];

            let mut nthreads = 32;
            let limit = iter_dim.min(pipeline.max_total_threads_per_threadgroup() as usize);
            while (nthreads * 2) < limit {
                nthreads *= 2;
            }

            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                let mut idx = 0;
                encoder.set_metal_tensor(idx, input, metal::MTLResourceUsage::Read);
                idx += 1;
                if let Some(residual) = residual {
                    encoder.set_metal_tensor(idx, residual, metal::MTLResourceUsage::Read);
                    idx += 1;
                }
                encoder.set_tensor(idx, eps);
                idx += 1;
                if let Some(scale) = scale {
                    encoder.set_metal_tensor(idx, scale, metal::MTLResourceUsage::Read);
                    idx += 1;
                }
                encoder.set_metal_tensor(idx, output, metal::MTLResourceUsage::Write);
                idx += 1;
                if let Some(sum_out) = sum_out {
                    encoder.set_metal_tensor(idx, sum_out, metal::MTLResourceUsage::Write);
                    idx += 1;
                }
                encoder.set_slice(idx, &shape_nd3);
                encoder.set_slice(idx + 1, &strides_nd3);
                encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
                let grid_size =
                    MTLSize { width: (shape_nd3[2] * shape_nd3[0]) as _, height: 1, depth: 1 };
                let group_size = MTLSize { width: nthreads as _, height: 1, depth: 1 };

                encoder.dispatch_thread_groups(grid_size, group_size);
            });
        }
        Ok(())
    }
}

pub fn metal_rms_norm_dispatch(
    input: &DeviceTensor,
    residual: Option<&DeviceTensor>,
    scale: Option<&DeviceTensor>,
    axis: usize,
    eps: &Tensor,
    output: &DeviceTensor,
    sum_out: Option<&DeviceTensor>,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        RmsNorm.dispatch_eval(stream, input, residual, scale, axis, eps, output, sum_out)
    })
}

crate::register_metal_op!(tract_transformers::ops::rms_norm::RmsNorm, |source, node, op| {
    rule_if!(RmsNorm::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
        op.axis,
        op.eps.clone(),
        false,
        false,
        None,
        "Metal",
        metal_rms_norm_dispatch,
    ))))
});

crate::register_metal_op!(
    tract_core::ops::nn::ScaledRmsNorm,
    |source, node, op| {
        let input_facts = source.node_input_facts(node.id)?;
        rule_if!(RmsNorm::is_supported_dt(input_facts[0].datum_type));
        rule_if!(op.out_dt.map(RmsNorm::is_supported_dt).unwrap_or(true));
        rule_if!(input_facts[1].datum_type == DatumType::F32);
        Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
            op.axis,
            op.eps.clone(),
            true,
            false,
            op.out_dt,
            "Metal",
            metal_rms_norm_dispatch,
        ))))
    }
);

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;

    use super::*;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;
    use tract_transformers::ops::rms_norm;

    fn test_case<F>(shape: &[usize], axis: usize, offset: f32, scale: f32) -> TractResult<()>
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

            let eps = Arc::new(tensor0(0.0001f32));
            let cpu_rms = rms_norm::RmsNorm { axis, eps: Arc::clone(&eps) };

            let cpu_output = cpu_rms
                .eval(&EvalContext::out_of_plan(), tvec![a.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_output = RmsNorm.eval(stream, &a, None, None, axis, &eps)?;

            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)
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
    fn test_rms() -> TractResult<()> {
        test_case::<f32>(&[4, 4], 1, -8.0, 1.0 / 100.0)?;
        test_case::<f16>(&[4, 4], 1, -8.0, 1.0 / 100.0)?;
        Ok(())
    }

    fn scaled_test_case<F>(shape: &[usize], axis: usize, out_dt: DatumType) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        use tract_core::ops::nn::ScaledRmsNorm;
        with_borrowed_metal_stream(|stream| {
            let len = shape.iter().product::<usize>();
            let dim = shape[axis];

            let input = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v / 33.0 - 5.0).as_()
                    })
                    .collect::<Vec<_>>(),
            )?;
            let scale = Tensor::from_shape(
                &[dim],
                &(0..dim).map(|f| -> f32 { 0.5 + (f as f32) / dim as f32 }).collect::<Vec<_>>(),
            )?;

            let eps = Arc::new(tensor0(0.0001f32));
            let cpu_op =
                ScaledRmsNorm { axis, eps: Arc::clone(&eps), out_dt: Some(out_dt) };
            let cpu_output = cpu_op
                .eval(
                    &EvalContext::out_of_plan(),
                    tvec![input.clone().into_tvalue(), scale.clone().into_tvalue()],
                )?[0]
                .clone()
                .into_tensor();

            let input_m = input.into_device()?;
            let scale_m = scale.into_device()?;
            let metal_output =
                RmsNorm.eval(stream, &input_m, Some(&scale_m), Some(out_dt), axis, &eps)?;
            let metal_output = metal_output.to_host()?.into_tensor();

            ensure!(metal_output.datum_type() == out_dt);
            cpu_output
                .close_enough(&metal_output, Approximation::Approximate)
                .with_context(|| format!("Cpu: {cpu_output:?}, Metal: {metal_output:?}"))?;
            Ok(())
        })
    }

    #[test]
    fn test_rms_scaled_and_out_dt() -> TractResult<()> {
        // l4 fast path (last axis, multiple of 4) and nd3 path (non-last axis
        // or dim not multiple of 4), all in/out dtype combos.
        for (shape, axis) in [(&[6usize, 8][..], 1), (&[6, 9][..], 1), (&[8, 5][..], 0)] {
            scaled_test_case::<f32>(shape, axis, DatumType::F32)?;
            scaled_test_case::<f32>(shape, axis, DatumType::F16)?;
            scaled_test_case::<f16>(shape, axis, DatumType::F16)?;
            scaled_test_case::<f16>(shape, axis, DatumType::F32)?;
        }
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn rms_prop_f32(pb in any::<RmsNormProblem<f32>>()) {
            fn run(pb: RmsNormProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn rms_prop_f16(pb in any::<RmsNormProblem<f16>>()) {
            fn run(pb: RmsNormProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
                    let shape_len = usize::min(left + right, 4);
                    let iter_ax_dim = 1usize..1024;
                    let other_dim = 1usize..10;
                    (iter_ax_dim, vec(other_dim, shape_len..=shape_len), Just(axis))
                })
                .prop_map(|(iter_dim, mut shape, axis)| {
                    shape.insert(axis, iter_dim);
                    let input = (0..shape.iter().product::<usize>())
                        .map(|f| f.as_() / 1000.as_())
                        .collect::<Vec<_>>();
                    Self { shape, axis, input, eps: Arc::new(tensor0(0.0001f32)) }
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
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_rms = rms_norm::RmsNorm { axis: self.axis, eps: Arc::clone(&self.eps) };

            let cpu_output = cpu_rms.eval(&EvalContext::out_of_plan(), tvec![a.into_tvalue()])?[0]
                .clone()
                .into_tensor();

            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            with_borrowed_metal_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let metal_output = RmsNorm.eval(stream, &a, None, None, self.axis, &self.eps)?;
                Ok(metal_output.to_host()?.into_tensor())
            })
        }
    }
}

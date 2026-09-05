use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, MAX_THREADS, WARP_SIZE, get_cuda_view, utils};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
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
        n_cols: usize,
        scaled: bool,
        with_residual: bool,
    ) -> TractResult<String> {
        ensure!(Self::is_supported_dt(in_dt), "Unsupported dt {:?} for cuda rmsop", in_dt);
        ensure!(Self::is_supported_dt(out_dt), "Unsupported dt {:?} for cuda rmsop", out_dt);
        let iname = DeviceTensor::tname(in_dt)?;
        let oname = DeviceTensor::tname(out_dt)?;
        let variant = match (scaled, with_residual) {
            (true, true) => "rms_norm_scaled_add",
            (true, false) => "rms_norm_scaled",
            (false, true) => "rms_norm_add",
            (false, false) => "rms_norm",
        };
        if n_cols < MAX_THREADS {
            Ok(format!("{variant}_small_{iname}_{oname}"))
        } else {
            Ok(format!("{variant}_{iname}_{oname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        scale: Option<&DeviceTensor>,
        axis: usize,
        eps: &Tensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, None, scale, axis, eps, &output, None)?;
        stream.synchronize()?;
        Ok(output)
    }

    /// `residual`/`sum_out` come and go together, same contract as the
    /// Metal kernel: the normalized value is `input + residual` (computed
    /// in the input dtype, bit-identical to a standalone Add), also written
    /// to `sum_out` for the residual stream's other consumers.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        residual: Option<&DeviceTensor>,
        scale: Option<&DeviceTensor>,
        axis: usize,
        eps: &Tensor,
        output: &DeviceTensor,
        sum_out: Option<&DeviceTensor>,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(Self::is_supported_dt(output.datum_type()));
        ensure!(residual.is_some() == sum_out.is_some());
        if let (Some(residual), Some(sum_out)) = (residual, sum_out) {
            ensure!(residual.shape() == input.shape());
            ensure!(residual.datum_type() == input.datum_type());
            ensure!(sum_out.shape() == input.shape());
            ensure!(sum_out.datum_type() == input.datum_type());
        }
        if let Some(scale) = scale {
            ensure!(scale.datum_type() == DatumType::F32);
            ensure!(scale.len() == input.shape()[axis]);
        }

        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        let kernel_name = self.kernel_name(
            input.datum_type(),
            output.datum_type(),
            shape_nd3[1],
            scale.is_some(),
            residual.is_some(),
        )?;

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        // Unused optional slots are never dereferenced on the device side
        // (the kernel's `HAS_RESIDUAL`/`HAS_SCALE` template flags eliminate
        // that code at compile time), so any valid view is a safe
        // placeholder -- reuse the output's.
        let r_view = residual.map(get_cuda_view).unwrap_or_else(|| get_cuda_view(output));
        let s_view = scale.map(get_cuda_view).unwrap_or_else(|| get_cuda_view(output));
        let sum_view = sum_out.map(get_cuda_view).unwrap_or_else(|| get_cuda_view(output));

        let func = cuda_context().load_pipeline(LibraryName::NN, kernel_name)?;
        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&i_view);
        launch_args.push_view(&r_view);
        launch_args.push_view(&s_view);
        launch_args.push_view(&o_view);
        launch_args.push_view(&sum_view);
        launch_args.push_slice_i32(&shape_nd3);
        launch_args.push_slice_i32(&strides_nd3);
        launch_args.push::<f32>(*eps.try_as_plain()?.to_scalar::<f32>()?);

        let cfg = LaunchConfig {
            grid_dim: ((shape_nd3[2] * shape_nd3[0]) as _, 1, 1),
            block_dim: if shape_nd3[1] < MAX_THREADS {
                (WARP_SIZE as _, 1, 1)
            } else {
                (MAX_THREADS as _, 1, 1)
            },
            shared_mem_bytes: 0,
        };

        launch_args.launch(cfg)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cuda_rms_norm_dispatch(
    input: &DeviceTensor,
    residual: Option<&DeviceTensor>,
    scale: Option<&DeviceTensor>,
    axis: usize,
    eps: &Tensor,
    output: &DeviceTensor,
    sum_out: Option<&DeviceTensor>,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        RmsNorm.dispatch_eval(stream, input, residual, scale, axis, eps, output, sum_out)
    })
}

crate::register_cuda_op!(tract_transformers::ops::rms_norm::RmsNorm, |source, node, op| {
    rule_if!(RmsNorm::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::rms_norm::GpuRmsNorm::new(
        op.axis,
        op.eps.clone(),
        false,
        false,
        None,
        "Cuda",
        cuda_rms_norm_dispatch,
    ))))
});

crate::register_cuda_op!(tract_core::ops::nn::ScaledRmsNorm, |source, node, op| {
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
        "Cuda",
        cuda_rms_norm_dispatch,
    ))))
});

#[cfg(test)]
mod tests {
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
        crate::with_cuda_stream(|stream| {
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
            let cuda_output = RmsNorm.eval(stream, &a, None, axis, &eps)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)
                .with_context(|| {
                    format!(
                        "Input: {:?}, Cpu: {:?}, Cuda: {:?}",
                        a.to_host().and_then(|it| it.dump(true)),
                        cpu_output.dump(true),
                        cuda_output.to_host().and_then(|it| it.dump(true))
                    )
                })?;
            Ok(())
        })
    }

    #[test]
    fn test_rms() -> TractResult<()> {
        test_case::<f32>(&[2, 2], 1, -0.0, 1.0 / 100.0)?;
        test_case::<f16>(&[2, 7], 0, -0.0, 1.0 / 100.0)?;
        test_case::<f32>(&[2, 124], 1, -0.0, 1.0 / 100.0)?;
        test_case::<f16>(&[1026, 7], 0, -0.0, 1.0 / 100.0)?;
        Ok(())
    }

    fn scaled_test_case<F>(shape: &[usize], axis: usize) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        use tract_core::ops::nn::ScaledRmsNorm;
        crate::with_cuda_stream(|stream| {
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
            let cpu_op = ScaledRmsNorm { axis, eps: Arc::clone(&eps), out_dt: None };
            let cpu_output = cpu_op.eval(
                &EvalContext::out_of_plan(),
                tvec![input.clone().into_tvalue(), scale.clone().into_tvalue()],
            )?[0]
                .clone()
                .into_tensor();

            let input_c = input.into_device()?;
            let scale_c = scale.into_device()?;
            let cuda_output = RmsNorm.eval(stream, &input_c, Some(&scale_c), axis, &eps)?;
            let cuda_output = cuda_output.to_host()?.into_tensor();

            cpu_output
                .close_enough(&cuda_output, Approximation::Approximate)
                .with_context(|| format!("Cpu: {cpu_output:?}, Cuda: {cuda_output:?}"))?;
            Ok(())
        })
    }

    #[test]
    fn test_rms_scaled() -> TractResult<()> {
        // "small" (< MAX_THREADS) and the regular reduction path, both dtypes.
        for (shape, axis) in [(&[6usize, 8][..], 1), (&[6, 9][..], 1), (&[8, 5][..], 0)] {
            scaled_test_case::<f32>(shape, axis)?;
            scaled_test_case::<f16>(shape, axis)?;
        }
        scaled_test_case::<f32>(&[2, 1200], 1)?;
        scaled_test_case::<f16>(&[2, 1200], 1)?;
        Ok(())
    }

    /// The OpenELM QK-norm shape: an f16 activation normalized (and scaled)
    /// into an f32 buffer for a higher-precision downstream op -- the
    /// specific in/out dtype mismatch that used to make the Cuda
    /// registration decline `ScaledRmsNorm` and fall back to the CPU op.
    fn cast_test_case(
        shape: &[usize],
        axis: usize,
        in_dt: DatumType,
        out_dt: DatumType,
    ) -> TractResult<()> {
        use tract_core::ops::nn::ScaledRmsNorm;
        crate::with_cuda_stream(|stream| {
            let len = shape.iter().product::<usize>();
            let dim = shape[axis];

            let input = Tensor::from_shape(
                shape,
                &(0..len).map(|f| (f as f32) / 33.0 - 5.0).collect::<Vec<_>>(),
            )?
            .cast_to_dt(in_dt)?
            .into_owned();
            let scale = Tensor::from_shape(
                &[dim],
                &(0..dim).map(|f| 0.5 + (f as f32) / dim as f32).collect::<Vec<_>>(),
            )?;

            let eps = Arc::new(tensor0(0.0001f32));
            let cpu_op = ScaledRmsNorm { axis, eps: Arc::clone(&eps), out_dt: Some(out_dt) };
            let cpu_output = cpu_op.eval(
                &EvalContext::out_of_plan(),
                tvec![input.clone().into_tvalue(), scale.clone().into_tvalue()],
            )?[0]
                .clone()
                .into_tensor();

            let input_c = input.into_device()?;
            let scale_c = scale.into_device()?;
            let output_c = unsafe { DeviceTensor::uninitialized_dt(out_dt, input_c.shape())? };
            RmsNorm.dispatch_eval(
                stream,
                &input_c,
                None,
                Some(&scale_c),
                axis,
                &eps,
                &output_c,
                None,
            )?;
            stream.synchronize()?;
            let cuda_output = output_c.to_host()?.into_tensor();

            cpu_output
                .close_enough(&cuda_output, Approximation::Approximate)
                .with_context(|| format!("Cpu: {cpu_output:?}, Cuda: {cuda_output:?}"))?;
            Ok(())
        })
    }

    #[test]
    fn test_rms_scaled_cast() -> TractResult<()> {
        for (shape, axis) in [(&[6usize, 8][..], 1), (&[6, 9][..], 1), (&[8, 5][..], 0)] {
            cast_test_case(shape, axis, DatumType::F16, DatumType::F32)?;
            cast_test_case(shape, axis, DatumType::F32, DatumType::F16)?;
        }
        cast_test_case(&[2, 1200], 1, DatumType::F16, DatumType::F32)?;
        cast_test_case(&[2, 1200], 1, DatumType::F32, DatumType::F16)?;
        Ok(())
    }

    fn residual_test_case<F>(shape: &[usize], axis: usize, with_scale: bool) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        use tract_core::ops::binary::TypedBinOp;
        use tract_core::ops::math::Add;
        use tract_core::ops::nn::ScaledRmsNorm;
        crate::with_cuda_stream(|stream| {
            let len = shape.iter().product::<usize>();
            let dim = shape[axis];

            let input = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v / 41.0 - 3.0).as_()
                    })
                    .collect::<Vec<_>>(),
            )?;
            let residual = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v / 19.0 - 1.0).as_()
                    })
                    .collect::<Vec<_>>(),
            )?;
            let scale = Tensor::from_shape(
                &[dim],
                &(0..dim).map(|f| -> f32 { 0.5 + (f as f32) / dim as f32 }).collect::<Vec<_>>(),
            )?;

            let eps = Arc::new(tensor0(0.0001f32));
            let sum = TypedBinOp(Box::new(Add), None).eval(
                &EvalContext::out_of_plan(),
                tvec![input.clone().into_tvalue(), residual.clone().into_tvalue()],
            )?[0]
                .clone()
                .into_tensor();
            let cpu_normed = if with_scale {
                ScaledRmsNorm { axis, eps: Arc::clone(&eps), out_dt: None }.eval(
                    &EvalContext::out_of_plan(),
                    tvec![sum.clone().into_tvalue(), scale.clone().into_tvalue()],
                )?[0]
                    .clone()
                    .into_tensor()
            } else {
                rms_norm::RmsNorm { axis, eps: Arc::clone(&eps) }
                    .eval(&EvalContext::out_of_plan(), tvec![sum.clone().into_tvalue()])?[0]
                    .clone()
                    .into_tensor()
            };

            let input_c = input.into_device()?;
            let residual_c = residual.into_device()?;
            let scale_c = scale.into_device()?;
            let output_c =
                unsafe { DeviceTensor::uninitialized_dt(input_c.datum_type(), input_c.shape())? };
            let sum_out_c =
                unsafe { DeviceTensor::uninitialized_dt(input_c.datum_type(), input_c.shape())? };
            RmsNorm.dispatch_eval(
                stream,
                &input_c,
                Some(&residual_c),
                with_scale.then_some(&scale_c),
                axis,
                &eps,
                &output_c,
                Some(&sum_out_c),
            )?;
            stream.synchronize()?;
            let cuda_normed = output_c.to_host()?.into_tensor();
            let cuda_sum = sum_out_c.to_host()?.into_tensor();

            cpu_normed
                .close_enough(&cuda_normed, Approximation::Approximate)
                .with_context(|| format!("normed: Cpu: {cpu_normed:?}, Cuda: {cuda_normed:?}"))?;
            sum.close_enough(&cuda_sum, Approximation::Approximate)
                .with_context(|| format!("sum: Cpu: {sum:?}, Cuda: {cuda_sum:?}"))?;
            Ok(())
        })
    }

    #[test]
    fn test_rms_residual() -> TractResult<()> {
        for (shape, axis) in [(&[6usize, 8][..], 1), (&[8, 5][..], 0), (&[2, 1200][..], 1)] {
            residual_test_case::<f32>(shape, axis, false)?;
            residual_test_case::<f16>(shape, axis, false)?;
            residual_test_case::<f32>(shape, axis, true)?;
            residual_test_case::<f16>(shape, axis, true)?;
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
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn rms_prop_f16(pb in any::<RmsNormProblem<f16>>()) {
            fn run(pb: RmsNormProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
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
            (0usize..5, 0usize..1)
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
            crate::with_cuda_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let cuda_output = RmsNorm.eval(stream, &a, None, self.axis, &self.eps)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

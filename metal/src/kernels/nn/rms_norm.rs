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

    pub fn kernel_name(&self, dt: DatumType, is_l4: bool) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal rms  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        if !is_l4 {
            Ok(format!("nn_ops::rms_norm_nd3_{tname}"))
        } else {
            Ok(format!("nn_ops::rms_norm_nd2_l4_{tname}"))
        }
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axis: usize,
        eps: &Tensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, axis, eps, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axis: usize,
        eps: &Tensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        if (axis == (input.rank() - 1)) && (input.shape()[axis] % 4 == 0) {
            let shape = input.shape();
            let shape_nd2 = tvec![shape[..axis].iter().product::<usize>(), shape[axis]];

            let pipeline = stream
                .load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type(), true)?)?;

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
                encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                encoder.set_tensor(1, eps);
                encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<usize>() as u64,
                    &iter_dim as *const usize as *const _,
                );
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<usize>() as u64,
                    &iter_dim_div_4 as *const usize as *const _,
                );
                encoder.set_bytes(
                    5,
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

            let pipeline = stream
                .load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type(), false)?)?;

            let iter_dim = shape_nd3[1];

            let mut nthreads = 32;
            let limit = iter_dim.min(pipeline.max_total_threads_per_threadgroup() as usize);
            while (nthreads * 2) < limit {
                nthreads *= 2;
            }

            let command_buffer = stream.command_buffer();
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                encoder.set_tensor(1, eps);
                encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
                encoder.set_slice(3, &shape_nd3);
                encoder.set_slice(4, &strides_nd3);
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

            let eps = Arc::new(tensor0(0.0001f32.as_()));
            let cpu_rms = rms_norm::RmsNorm { axis, eps: Arc::clone(&eps) };

            let cpu_output =
                cpu_rms.eval(tvec![a.to_host()?.into_tvalue()])?[0].clone().into_tensor();
            let metal_output = RmsNorm.eval(stream, &a, axis, &eps)?;

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
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;

            let cpu_rms = rms_norm::RmsNorm { axis: self.axis, eps: Arc::clone(&self.eps) };

            let cpu_output = cpu_rms.eval(tvec![a.into_tvalue()])?[0].clone().into_tensor();

            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            with_borrowed_metal_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let metal_output = RmsNorm.eval(stream, &a, self.axis, &self.eps)?;
                Ok(metal_output.to_host()?.into_tensor())
            })
        }
    }
}

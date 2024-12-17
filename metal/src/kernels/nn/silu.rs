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
            let group_size = MTLSize {
                width: pipeline.max_total_threads_per_threadgroup(),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, group_size);

            encoder.end_encoding();
        });
        Ok(())
    }

    pub fn dispatch_eval_leg(
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
    use std::time::Duration;
    use std::time::Instant;

    use super::*;
    use crate::rewrite_rules::BasicSilu;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;
    use tract_smallvec::SmallVec;

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
        test_case::<f32>(&[4, 6], -0.0, 1.0 / 100.0, Approximation::Approximate)?;
        test_case::<f16>(&[4, 6], -6.0, 1.0 / 1000.0, Approximation::SuperApproximate)?;
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

    #[test]
    pub fn profile_silu() -> Result<()> {
        let n_iters = 1000;
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let shape = &[4096, 1536];
                let len = shape.iter().product::<usize>();

                let mut rng = rand::thread_rng();
                let a = Tensor::from_shape(
                    shape,
                    &(0..len)
                        .map(|_| -> f32 {
                            let v: u64 = rng.gen_range(0..100);
                            v as f32 * 1. - 50.
                        })
                        .collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let output = unsafe { MetalTensor::uninitialized_dt(a.datum_type(), a.shape())? };
                for _ in 0..100 {
                    Silu.dispatch_eval(context, &a, &output)?;
                }
                context.wait_until_completed()?;

                dbg!(a.shape());
                println!("Running custom  dispatch");
                let (mut cpu_start, mut gpu_start): (u64, u64) = (0, 0);
                context.device().sample_timestamps(&mut cpu_start, &mut gpu_start);

                let mut encod_time = Duration::default();
                let (_, total_dur, profile_gpu) = context.profile(n_iters, || {
                    let tot_dur = Instant::now();
                    for i in 0..n_iters {
                        if let Some(profiler) = context.profiler() {
                            profiler.borrow_mut().add_node_entry(i);
                        };

                        let start = Instant::now();
                        Silu.dispatch_eval(context, &a, &output)?;
                        encod_time += start.elapsed();
                    }

                    context.wait_until_completed()?;
                    Ok((SmallVec::new(), tot_dur.elapsed()))
                })?;

                let (mut cpu_end, mut gpu_end): (u64, u64) = (0, 0);
                context.device().sample_timestamps(&mut cpu_end, &mut gpu_end);

                let gpu_sum = Duration::from_nanos(crate::utils::rescale_gpu_duration(
                    profile_gpu.iter().sum(),
                    cpu_start,
                    cpu_end,
                    gpu_start,
                    gpu_end,
                ));

                let encoding_time_per_iter = encod_time.div_f32(n_iters as f32).as_micros();
                let tot_dur_per_iter = total_dur.div_f32(n_iters as f32).as_micros();
                let gpu_per_iter = gpu_sum.div_f32(n_iters as f32).as_micros();

                println!("CPU duration: {tot_dur_per_iter:?} us");
                println!("  dont Encoding time: {encoding_time_per_iter:?} us");
                println!("GPU duration: {gpu_per_iter:?} us");

                println!("\nRunning legacy dispatch");
                let (mut cpu_start, mut gpu_start): (u64, u64) = (0, 0);
                context.device().sample_timestamps(&mut cpu_start, &mut gpu_start);

                let mut encod_time = Duration::default();
                let (_, total_dur, profile_gpu) = context.profile(n_iters, || {
                    let tot_dur = Instant::now();
                    for i in 0..n_iters {
                        if let Some(profiler) = context.profiler() {
                            profiler.borrow_mut().add_node_entry(i);
                        };

                        let start = Instant::now();
                        Silu.dispatch_eval_leg(context, &a, &output)?;
                        encod_time += start.elapsed();
                    }

                    context.wait_until_completed()?;
                    Ok((SmallVec::new(), tot_dur.elapsed()))
                })?;

                let (mut cpu_end, mut gpu_end): (u64, u64) = (0, 0);
                context.device().sample_timestamps(&mut cpu_end, &mut gpu_end);

                let gpu_sum = Duration::from_nanos(crate::utils::rescale_gpu_duration(
                    profile_gpu.iter().sum(),
                    cpu_start,
                    cpu_end,
                    gpu_start,
                    gpu_end,
                ));

                let encoding_time_per_iter = encod_time.div_f32(n_iters as f32).as_micros();
                let tot_dur_per_iter_leg = total_dur.div_f32(n_iters as f32).as_micros();
                let gpu_per_iter_leg = gpu_sum.div_f32(n_iters as f32).as_micros();

                println!("CPU duration: {tot_dur_per_iter_leg:?} us");
                println!("  dont Encoding time: {encoding_time_per_iter:?} us");
                println!("GPU duration: {gpu_per_iter_leg:?} us");

                let flat_gain = tot_dur_per_iter_leg as f32 - tot_dur_per_iter as f32;
                let flat_ratio = flat_gain * 100. / tot_dur_per_iter_leg as f32;
                let flat_gain_gpu = gpu_per_iter_leg as f32 - gpu_per_iter as f32;
                let flat_ratio_gpu = flat_gain_gpu * 100. / gpu_per_iter_leg as f32;
                println!("\nTotal Flat gain: {flat_gain} us -> {flat_ratio} %");
                println!("GPU Flat gain: {flat_gain_gpu} us -> {flat_ratio_gpu} %");
                Ok(())
            })
        })
    }
}

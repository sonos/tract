use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view, launch_args, utils};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_core::ops::nn::SoftmaxKind;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Softmax;

impl Softmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(
        &self,
        dt: DatumType,
        n_cols: usize,
        kind: SoftmaxKind,
    ) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda softmaxop", dt);
        let tname = DeviceTensor::tname(dt)?;
        let op = match kind {
            SoftmaxKind::Softmax => "softmax",
            SoftmaxKind::LogSoftmax => "log_softmax",
        };
        let size = if n_cols < MAX_THREADS { "small_" } else { "" };
        Ok(format!("{op}_{size}{tname}"))
    }

    pub fn split_kernel_names(dt: DatumType) -> TractResult<[String; 3]> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda softmaxop", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok([
            format!("softmax_split_partial_{tname}"),
            format!("softmax_split_finalize_{tname}"),
            format!("log_softmax_split_finalize_{tname}"),
        ])
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        axis: usize,
        kind: SoftmaxKind,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, axis, kind, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        axis: usize,
        kind: SoftmaxKind,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let shape_nd3 = utils::reshape_to_rank_3(input.shape(), axis);
        let strides_nd3 = Tensor::natural_strides(&shape_nd3);

        let rows = shape_nd3[0] * shape_nd3[2];
        if shape_nd3[2] == 1
            && let Some(chunk) = split_chunk(rows, shape_nd3[1])
        {
            return self.dispatch_split(stream, input, kind, output, rows, shape_nd3[1], chunk);
        }

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let func = cuda_context().load_pipeline(
            LibraryName::NN,
            self.kernel_name(input.datum_type(), shape_nd3[1], kind)?,
        )?;
        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&i_view);
        launch_args.push_view(&o_view);
        launch_args.push_slice_i32(&shape_nd3);
        launch_args.push_slice_i32(&strides_nd3);

        let cfg = LaunchConfig {
            grid_dim: ((shape_nd3[0] * shape_nd3[2]) as _, 1, 1),
            block_dim: if shape_nd3[1] < MAX_THREADS {
                (32, 1, 1)
            } else {
                (MAX_THREADS as _, 1, 1)
            },
            shared_mem_bytes: 0,
        };

        launch_args.launch(cfg)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_split(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        kind: SoftmaxKind,
        output: &DeviceTensor,
        rows: usize,
        k: usize,
        chunk: usize,
    ) -> TractResult<()> {
        let [partial_name, softmax_name, log_softmax_name] =
            Self::split_kernel_names(input.datum_type())?;
        let splits = k.div_ceil(chunk);
        let part = DeviceTensor::uninitialized_dt(DatumType::F32, &[rows, splits, 2])?;
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        let p_view = get_cuda_view(&part);
        let cfg = LaunchConfig {
            grid_dim: (splits as _, rows as _, 1),
            block_dim: (MAX_THREADS as _, 1, 1),
            shared_mem_bytes: 0,
        };

        let partial = cuda_context().load_pipeline(LibraryName::NN, partial_name)?;
        let mut args = TractLaunchArgs::new(stream, &partial);
        args.push_view(&i_view);
        args.push_view(&p_view);
        args.push_i32(k);
        args.push_i32(chunk);
        args.launch(cfg)?;

        let finalize_name = match kind {
            SoftmaxKind::Softmax => softmax_name,
            SoftmaxKind::LogSoftmax => log_softmax_name,
        };
        let finalize = cuda_context().load_pipeline(LibraryName::NN, finalize_name)?;
        let mut args = TractLaunchArgs::new(stream, &finalize);
        args.push_view(&i_view);
        args.push_view(&o_view);
        args.push_view(&p_view);
        args.push_i32(k);
        args.push_i32(chunk);
        args.launch(cfg)
    }
}

/// The chunk length when a last-axis softmax splits each row over several blocks: with too few
/// rows to give every SM one, a row spreads over enough blocks to make about one per SM, each of
/// at least `SPLIT_MIN_CHUNK` elements. `None` keeps a block per row.
fn split_chunk(rows: usize, k: usize) -> Option<usize> {
    let sms = cuda_context().properties().multiProcessorCount as usize;
    let splits = (sms / rows).min(k / SPLIT_MIN_CHUNK);
    (splits >= 2).then(|| k.div_ceil(splits))
}

const SPLIT_MIN_CHUNK: usize = 2048;

pub fn cuda_softmax_dispatch(
    input: &DeviceTensor,
    axis: usize,
    kind: SoftmaxKind,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| Softmax.dispatch_eval(stream, input, axis, kind, output))
}

crate::register_cuda_op!(tract_core::ops::nn::Softmax, |source, node, op| {
    rule_if!(Softmax::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    rule_if!(op.axes.len() == 1);
    Ok(Some(Box::new(tract_gpu::ops::softmax::GpuSoftmax::from_tract_core(
        op,
        "Cuda",
        cuda_softmax_dispatch,
    )?)))
});

#[cfg(test)]
mod tests {

    use super::*;
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
        crate::with_cuda_stream(|stream| {
            let m = 2;
            let k = 3;
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
            let cuda_output = Softmax.eval(stream, &a, axis, SoftmaxKind::Softmax)?;

            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_f32_2() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
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
            let cuda_output = Softmax.eval(stream, &a, axis, SoftmaxKind::Softmax)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_softmax_f16() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
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
            let cuda_output = Softmax.eval(stream, &a, axis, SoftmaxKind::Softmax)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_log_softmax_large_f32() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            let shape = [3, 2000];
            let input = (0..6000).map(|f| ((f * 37) % 101) as f32 / 10.0).collect::<Vec<_>>();
            let a = Tensor::from_shape(&shape, &input)?.into_device()?;
            let cpu_softmax = TractSoftmax {
                axes: tvec![1],
                quant_output_dt: None,
                kind: SoftmaxKind::LogSoftmax,
            };
            let cpu_output = cpu_softmax
                .eval(&EvalContext::out_of_plan(), tvec![a.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let cuda_output = Softmax.eval(stream, &a, 1, SoftmaxKind::LogSoftmax)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)
        })
    }

    #[test]
    fn test_softmax_masked_f32() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            for kind in [SoftmaxKind::Softmax, SoftmaxKind::LogSoftmax] {
                for k in [7, 300, 5000, 20001, 70000] {
                    let mut input =
                        (0..3 * k).map(|f| ((f * 37) % 101) as f32 / 10.0).collect::<Vec<_>>();
                    for r in 0..3 {
                        input[r * k + (r * 5) % k] = f32::NEG_INFINITY;
                        input[r * k] = f32::NEG_INFINITY;
                    }
                    input[..k / 2].iter_mut().for_each(|x| *x = f32::NEG_INFINITY);
                    let a = Tensor::from_shape(&[3, k], &input)?.into_device()?;
                    let cpu = TractSoftmax { axes: tvec![1], quant_output_dt: None, kind }
                        .eval(&EvalContext::out_of_plan(), tvec![a.to_host()?.into_tvalue()])?[0]
                        .clone()
                        .into_tensor();
                    let cuda = Softmax.eval(stream, &a, 1, kind)?;
                    cpu.close_enough(&cuda.to_host()?.into_tensor(), Approximation::Approximate)
                        .with_context(|| format!("{kind:?} k={k}"))?;
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
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn softmax_prop_f16(pb in any::<SoftmaxProblem<f16>>()) {
            fn run(pb: SoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
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
            crate::with_cuda_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let cuda_output = Softmax.eval(stream, &a, self.axis, self.kind)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

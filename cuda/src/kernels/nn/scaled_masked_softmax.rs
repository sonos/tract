use std::iter::repeat_n;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view, launch_args};
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use num_traits::AsPrimitive;
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaledMaskedSoftmax;

impl ScaledMaskedSoftmax {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn is_supported_mask_dt(input_dt: DatumType, mask_dt: DatumType) -> bool {
        mask_dt == input_dt || mask_dt == bool::datum_type()
    }

    pub fn kernel_name(
        &self,
        input_dt: DatumType,
        mask_is_bool: bool,
        block_size: usize,
    ) -> TractResult<String> {
        ensure!(
            Self::is_supported_dt(input_dt),
            "Unsupported dt {:?} for cuda scaled masked softmax op",
            input_dt
        );
        let tname = DeviceTensor::tname(input_dt)?;
        let stem =
            if mask_is_bool { "scaled_bool_masked_softmax" } else { "scaled_masked_softmax" };
        Ok(format!("{stem}_{block_size}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
        post_softmax_mask: bool,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, scale, mask, post_softmax_mask, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
        post_softmax_mask: bool,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(input.rank() >= 2 && input.rank() <= 5);
        ensure!(mask.rank() == input.rank());
        ensure!(output.datum_type() == input.datum_type());
        let mask_is_bool = mask.datum_type() == bool::datum_type();
        ensure!(Self::is_supported_mask_dt(input.datum_type(), mask.datum_type()));
        // post_softmax_mask is meaningful only with a bool mask (CPU contract).
        ensure!(!post_softmax_mask || mask_is_bool);

        let shape = pad(input.shape(), 1);
        let strides = pad(input.strides(), 0);
        let mask_strides = pad(&compute_broadcast_strides::<i32>(mask.shape(), mask.strides())?, 0);
        let output_strides = pad(output.strides(), 0);

        let i_view = get_cuda_view(input);
        let mask_view = get_cuda_view(mask);
        let o_view = get_cuda_view(output);

        let inner_len = shape[4] as usize;
        let mut nth = 32;
        while nth < inner_len && nth < MAX_THREADS {
            nth *= 2;
        }

        let block_size =
            if inner_len.is_power_of_two() && inner_len > 32 { inner_len.min(1024) } else { 0 };

        let func = cuda_context().load_pipeline(
            LibraryName::NN,
            self.kernel_name(input.datum_type(), mask_is_bool, block_size)?,
        )?;

        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&i_view);
        launch_args.push_view(&mask_view);
        launch_args.push::<f32>(scale.cast_to_scalar::<f32>()?);
        launch_args.push_view(&o_view);
        if mask_is_bool {
            launch_args.push::<i32>(post_softmax_mask as i32);
        }
        launch_args.push_slice_i32(&shape);
        launch_args.push_slice_i32(&strides);
        launch_args.push_slice_i32(&mask_strides);
        launch_args.push_slice_i32(&output_strides);

        // input is [b, kh, gh, row, col]
        // grid_dim= (row, gh, b*kh)
        let cfg = LaunchConfig {
            grid_dim: (shape[3] as _, shape[2] as _, (shape[0] * shape[1]) as _),
            block_dim: (nth as _, 1, 1),
            shared_mem_bytes: ((inner_len.next_power_of_two() + 32) * size_of::<f32>()) as u32,
        };

        launch_args.launch(cfg)
    }
}

fn pad(vals: &[impl AsPrimitive<i32>], neutral: i32) -> [i32; 5] {
    let mut it = [neutral; 5];
    for (ix, val) in vals.iter().enumerate() {
        it[ix + 5 - vals.len()] = val.as_();
    }
    it
}

pub fn cuda_scaled_masked_softmax_dispatch(
    input: &DeviceTensor,
    scale: &Tensor,
    mask: &DeviceTensor,
    post_softmax_mask: bool,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        ScaledMaskedSoftmax.dispatch_eval(stream, input, scale, mask, post_softmax_mask, output)
    })
}

crate::register_cuda_op!(
    tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax,
    |source, node, op| {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(ScaledMaskedSoftmax::is_supported_dt(facts[0].datum_type));
        rule_if!(ScaledMaskedSoftmax::is_supported_mask_dt(
            facts[0].datum_type,
            facts[1].datum_type,
        ));
        // post_softmax_mask requires a bool mask (CPU contract).
        rule_if!(!op.post_softmax_mask || facts[1].datum_type == bool::datum_type());
        Ok(Some(Box::new(tract_gpu::ops::scaled_masked_softmax::GpuScaledMaskedSoftmax::new(
            op.scale.clone(),
            op.post_softmax_mask,
            "Cuda",
            cuda_scaled_masked_softmax_dispatch,
        ))))
    }
);

#[cfg(test)]
mod tests {
    use tract_gpu::tensor::IntoDevice;

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
    fn test_scaled_masked_softmax_f32() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            let m = 6;
            let n = 33;
            let scale: Arc<_> = tensor0(0.125f32).into();
            let mask = Tensor::from_shape(&[1, 1, m, n], &vec![-1000f32; m * n])?.into_device()?;

            let a = Tensor::from_shape(
                &[4, 1, m, n],
                &(0..4 * m * n).map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cpu = scaled_masked_softmax::ScaledMaskedSoftmax {
                scale: scale.clone(),
                post_softmax_mask: false,
            };

            let cpu_output = cpu
                .eval(tvec![a.to_host()?.into_tvalue(), mask.to_host()?.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let cuda_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, false)?;
            cpu_output
                .close_enough(&cuda_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    /// Bool-mask path with a fully-masked row.  Without post_softmax_mask
    /// the output is NaN (matches CPU); with it on, the NaN is scrubbed to 0.
    #[test]
    fn test_scaled_bool_masked_softmax_post_mask_scrubs_nan() -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            let m = 3;
            let n = 5;
            let scale: Arc<_> = tensor0(0.125f32).into();
            // Row 0: fully masked.  Row 1: partially masked.  Row 2: fully unmasked.
            let mask_data: Vec<bool> = (0..m)
                .flat_map(|r| {
                    (0..n).map(move |c| match r {
                        0 => false,
                        1 => c >= 2,
                        _ => true,
                    })
                })
                .collect();
            let mask = Tensor::from_shape(&[1, 1, m, n], &mask_data)?.into_device()?;
            let a = Tensor::from_shape(
                &[1, 1, m, n],
                &(0..m * n).map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;

            for post in [false, true] {
                let cpu = scaled_masked_softmax::ScaledMaskedSoftmax {
                    scale: scale.clone(),
                    post_softmax_mask: post,
                };
                let cpu_out = cpu
                    .eval(tvec![a.to_host()?.into_tvalue(), mask.to_host()?.into_tvalue()])?[0]
                    .clone()
                    .into_tensor();
                let cuda_out = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, post)?;
                let cuda_host = cuda_out.to_host()?.into_tensor();
                let cpu_slice = cpu_out.view().as_slice::<f32>().unwrap();
                let cuda_slice = cuda_host.view().as_slice::<f32>().unwrap();
                for (i, (c, g)) in cpu_slice.iter().zip(cuda_slice.iter()).enumerate() {
                    if c.is_nan() {
                        assert!(g.is_nan(), "post={post} idx={i}: cpu NaN, cuda {g}");
                    } else {
                        assert!((c - g).abs() < 1e-5, "post={post} idx={i}: cpu {c} cuda {g}");
                    }
                }
            }
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
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn scaled_masked_softmax_prop_f16(pb in any::<ScaledMaskedSoftmaxProblem<f16>>()) {
            fn run(pb: ScaledMaskedSoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
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
            vec(1usize..10, 4..=4)
                .prop_map(|shape| {
                    let mut mask_shape = shape.clone();
                    mask_shape[0] = 1;
                    mask_shape[1] = 1;

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
        pub fn reference(&self) -> TractResult<Tensor> {
            let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?;
            let mask = Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?;
            let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();

            let cpu_output =
                scaled_masked_softmax::ScaledMaskedSoftmax { scale, post_softmax_mask: false }
                    .eval(tvec![a.into_tvalue(), mask.into_tvalue()])?[0]
                    .clone()
                    .into_tensor();
            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            crate::with_cuda_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let mask =
                    Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?.into_device()?;
                let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();
                let cuda_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, false)?;
                Ok(cuda_output.to_host()?.into_tensor())
            })
        }
    }
}

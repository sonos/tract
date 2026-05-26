use crate::encoder::EncoderExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use num_traits::AsPrimitive;
use tract_core::internal::*;
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

    pub fn kernel_name(&self, dt: DatumType, mask_is_bool: bool) -> TractResult<String> {
        ensure!(
            Self::is_supported_dt(dt),
            "Unsupported dt {:?} for metal scaled masked softmax op",
            dt
        );
        let tname = DeviceTensor::tname(dt)?;
        let stem = if mask_is_bool {
            "scaled_bool_masked_softmax_nd5"
        } else {
            "scaled_masked_softmax_nd5"
        };
        Ok(format!("nn_ops::{stem}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
        post_softmax_mask: bool,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, scale, mask, post_softmax_mask, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        scale: &Tensor,
        mask: &DeviceTensor,
        post_softmax_mask: bool,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(mask);
        stream.retain_tensor(output);

        ensure!(output.shape() == input.shape());
        ensure!(input.rank() >= 2);
        ensure!(input.rank() <= 5);
        ensure!(mask.rank() == input.rank());
        ensure!(output.datum_type() == input.datum_type());
        let mask_is_bool = mask.datum_type() == bool::datum_type();
        ensure!(Self::is_supported_mask_dt(input.datum_type(), mask.datum_type()));
        // post_softmax_mask is meaningful only with a bool mask (CPU contract).
        ensure!(!post_softmax_mask || mask_is_bool);
        let scale = scale.cast_to::<f32>()?;

        let shape = pad(input.shape(), 1);
        let strides = pad(input.strides(), 0);
        let mask_strides =
            pad(&compute_broadcast_strides::<usize>(mask.shape(), mask.strides())?, 0);
        let out_strides = pad(output.strides(), 0);

        let inner_len = shape[4] as usize;
        let mut nth = 32usize;
        while nth < inner_len && nth < 256 {
            nth *= 2;
        }

        let tg_floats = 32 + inner_len;
        let tg_bytes = tg_floats * f32::datum_type().size_of();

        let pipeline = stream.load_pipeline(
            LibraryName::NNOps,
            &self.kernel_name(input.datum_type(), mask_is_bool)?,
        )?;

        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, mask, metal::MTLResourceUsage::Read);
            encoder.set_tensor(2, &scale);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            // Bool-mask kernel takes a `post_mask` flag at slot 4; the
            // float-mask kernel doesn't, so slots shift down by one.
            let next_slot = if mask_is_bool {
                encoder.set_slice(4, &[post_softmax_mask as u32]);
                5
            } else {
                4
            };
            encoder.set_slice(next_slot, &shape);
            encoder.set_slice(next_slot + 1, &strides);
            encoder.set_slice(next_slot + 2, &mask_strides);
            encoder.set_slice(next_slot + 3, &out_strides);
            encoder.set_threadgroup_memory_length(0, tg_bytes as _);
            let grid_size = MTLSize {
                width: shape[3] as _,
                height: shape[2] as _,
                depth: (shape[0] * shape[1]) as _,
            };
            let group_size = MTLSize { width: nth as _, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

fn pad(vals: &[impl AsPrimitive<isize>], neutral: isize) -> [isize; 5] {
    let mut it = [neutral; 5];
    for (ix, val) in vals.iter().enumerate() {
        it[ix + 5 - vals.len()] = val.as_();
    }
    it
}

pub fn metal_scaled_masked_softmax_dispatch(
    input: &DeviceTensor,
    scale: &Tensor,
    mask: &DeviceTensor,
    post_softmax_mask: bool,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        ScaledMaskedSoftmax.dispatch_eval(stream, input, scale, mask, post_softmax_mask, output)
    })
}

crate::register_metal_op!(
    tract_transformers::ops::scaled_masked_softmax::ScaledMaskedSoftmax,
    |source, node, op| {
        let facts = source.node_input_facts(node.id)?;
        rule_if!(ScaledMaskedSoftmax::is_supported_dt(facts[0].datum_type));
        rule_if!(ScaledMaskedSoftmax::is_supported_mask_dt(
            facts[0].datum_type,
            facts[1].datum_type,
        ));
        rule_if!(!op.post_softmax_mask || facts[1].datum_type == bool::datum_type());
        Ok(Some(Box::new(tract_gpu::ops::scaled_masked_softmax::GpuScaledMaskedSoftmax::new(
            op.scale.clone(),
            op.post_softmax_mask,
            "Metal",
            metal_scaled_masked_softmax_dispatch,
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
    use proptest::strategy::Strategy;
    use tract_core::internal::Tensor;
    use tract_transformers::ops::scaled_masked_softmax;

    #[test]
    fn test_scaled_masked_softmax_f32() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let m = 4;
            let n = 4;
            let scale: Arc<_> = tensor0(0.125f32).into();
            let mask = Tensor::from_shape(&[1, 1, m, n], &vec![-1000f32; m * n])?.into_device()?;

            let a = Tensor::from_shape(
                &[1, 1, m, n],
                &(0..m * n).map(|f| f as f32).collect::<Vec<_>>(),
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
            let metal_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, false)?;
            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_scaled_masked_softmax_f32_2() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let m = 4;
            let n = 1024;
            let scale: Arc<_> = tensor0(0.125f32).into();
            let mask = Tensor::from_shape(&[1, 1, m, n], &vec![-1000f32; m * n])?.into_device()?;

            let a = Tensor::from_shape(
                &[1, 1, m, n],
                &(0..m * n).map(|f| f as f32).collect::<Vec<_>>(),
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
            let metal_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, false)?;
            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)?;
            Ok(())
        })
    }

    /// Bool-mask path with a fully-masked row.  Without post_softmax_mask
    /// the output is NaN (matches CPU); with it on, the NaN is scrubbed to 0.
    #[test]
    fn test_scaled_bool_masked_softmax_post_mask_scrubs_nan() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
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
                let metal_out = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, post)?;
                let metal_host = metal_out.to_host()?.into_tensor();
                let cpu_slice = cpu_out.view().as_slice::<f32>().unwrap();
                let metal_slice = metal_host.view().as_slice::<f32>().unwrap();
                for (i, (c, g)) in cpu_slice.iter().zip(metal_slice.iter()).enumerate() {
                    if c.is_nan() {
                        assert!(g.is_nan(), "post={post} idx={i}: cpu NaN, metal {g}");
                    } else {
                        assert!((c - g).abs() < 1e-5, "post={post} idx={i}: cpu {c} metal {g}");
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
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn scaled_masked_softmax_prop_f16(pb in any::<ScaledMaskedSoftmaxProblem<f16>>()) {
            fn run(pb: ScaledMaskedSoftmaxProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Metal: {:?}", reference.dump(true), out.dump(true)))
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
            with_borrowed_metal_stream(|stream| {
                let a = Tensor::from_shape(self.shape.as_slice(), &self.input)?.into_device()?;
                let mask =
                    Tensor::from_shape(self.mask_shape.as_slice(), &self.mask)?.into_device()?;
                let scale: Arc<_> = tensor0::<F>(0.125f32.as_()).into();
                let metal_output = ScaledMaskedSoftmax.eval(stream, &a, &scale, &mask, false)?;
                Ok(metal_output.to_host()?.into_tensor())
            })
        }
    }
}

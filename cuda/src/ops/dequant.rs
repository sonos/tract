use crate::kernels::block_dequant::{dispatch, is_q8_0};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q8_0};
use tract_gpu::tensor::DeviceTensorExt;

/// Expands a Q8_0 input into the plain floats the gemm already runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaBlockDequant;

impl Op for CudaBlockDequant {
    fn name(&self) -> StaticName {
        "CudaBlockDequant".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaBlockDequant {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = inputs[0].to_device_tensor()?;
        let output =
            tract_gpu::turn_handler::make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        if output.len() > 0 {
            crate::with_cuda_stream(|stream| dispatch(stream, input, &output))?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for CudaBlockDequant {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let input = facts[0];
            ensure!(is_q8_0(input), "CudaBlockDequant expected Q8_0, got {input:?}");
            ensure!(
                matches!(input.datum_type, DatumType::F16 | DatumType::F32),
                "Q8_0 dequant writes f16 or f32, got {:?}",
                input.datum_type
            );
            let k = input.shape.last().context("Q8_0 dequant input has no K")?.to_usize()?;
            let block_len = Q8_0.block_len();
            ensure!(
                k.is_multiple_of(block_len),
                "K {k} is not a multiple of the {block_len}-wide block"
            );
            Ok(tvec!(input.datum_type.fact(input.shape.clone())))
        })
        .with_context(|| "Error while computing facts for CudaBlockDequant")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaTransform;
    use std::sync::Arc;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::konst::Const;
    use tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0, Q8_0,
    };
    use tract_core::transform::ModelTransform;
    use tract_gpu::fact::DeviceTypedFactExt;

    fn quant_weight(format: &dyn BlockQuant, shape: &[usize], dt: DatumType) -> TractResult<Const> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (((i * 13 + 7) % 29) as f32 - 14.0) / 7.0).collect();
        let blob = format.quant_f32(&data)?;
        let m = shape[..shape.len() - 1].iter().product();
        let k = *shape.last().unwrap();
        let storage =
            BlockQuantStorage::new(tract_core::dyn_clone::clone_box(format), m, k, Arc::new(blob))?;
        let tensor = storage.into_tensor_with_shape(dt, shape);
        let fact =
            Box::new(BlockQuantFact::new(tract_core::dyn_clone::clone_box(format), shape.into()));
        Const::new_with_exotic_fact(tensor.into_arc_tensor(), fact)
    }

    fn matmul(act_dt: DatumType, weight: Const, m: usize, k: usize) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let a = model.add_source("a", act_dt.fact([m, k]))?;
        let w = model.wire_node("w", weight, &[])?[0];
        let mm = PrefixMatMul {
            transpose_a: false,
            transpose_b: true,
            transpose_c: false,
            quantize_output: None,
            operating_dt: None,
        };
        let out = model.wire_node("mm", mm, &[a, w])?;
        model.select_output_outlets(&out)?;
        Ok(model)
    }

    fn activation(m: usize, k: usize, dt: DatumType) -> TractResult<Tensor> {
        let n = m * k;
        let data: Vec<f32> = (0..n).map(|i| (((i * 3 + 1) % 17) as f32 - 8.0) / 8.0).collect();
        if dt == DatumType::F16 {
            let data: Vec<f16> = data.into_iter().map(f16::from_f32).collect();
            Tensor::from_shape(&[m, k], &data)
        } else {
            Tensor::from_shape(&[m, k], &data)
        }
    }

    fn op_names(model: &TypedModel) -> Vec<String> {
        model.nodes().iter().map(|n| n.op().name().to_string()).collect()
    }

    #[test]
    fn q8_0_matmul_matches_cpu_and_reaches_plain_gemm() -> TractResult<()> {
        for dt in [DatumType::F32, DatumType::F16] {
            let model = matmul(dt, quant_weight(&Q8_0, &[32, 256], dt)?, 4, 256)?;
            let input = activation(4, 256, dt)?;
            let cpu = model.clone().into_runnable()?.run(tvec![input.clone().into_tvalue()])?;

            let cuda_model = CudaTransform.transform_into(model)?;
            let names = op_names(&cuda_model);
            assert!(names.iter().any(|n| n == "CudaBlockDequant"), "{names:?}");
            let gemm = cuda_model
                .nodes()
                .iter()
                .find(|n| n.op().name().to_string().contains("Gemm"))
                .context("Q8_0 matmul did not reach a CUDA gemm")?;
            for fact in cuda_model.node_input_facts(gemm.id)? {
                assert!(fact.to_device_fact()?.is_plain(), "gemm input stayed quantized: {fact:?}");
            }
            let cuda = cuda_model.into_runnable()?.run(tvec![input.into_tvalue()])?;
            // Small M uses the existing matvec. f16 may accumulate in f16.
            let approx = if dt == DatumType::F16 {
                Approximation::VeryApproximate
            } else {
                Approximation::Approximate
            };
            cpu[0].clone().into_tensor().close_enough(&cuda[0].clone().into_tensor(), approx)?;
        }
        Ok(())
    }

    #[test]
    fn q4_0_stays_on_the_fused_gemm() -> TractResult<()> {
        let model = matmul(DatumType::F32, quant_weight(&Q4_0, &[8, 32], DatumType::F32)?, 4, 32)?;
        let input = activation(4, 32, DatumType::F32)?;
        let cpu = model.clone().into_runnable()?.run(tvec![input.clone().into_tvalue()])?;
        let cuda_model = CudaTransform.transform_into(model)?;
        let names = op_names(&cuda_model);
        assert!(names.iter().all(|n| n != "CudaBlockDequant"), "{names:?}");
        assert!(names.iter().any(|n| n.contains("Gemm")), "{names:?}");
        let cuda = cuda_model.into_runnable()?.run(tvec![input.into_tvalue()])?;
        cpu[0]
            .clone()
            .into_tensor()
            .close_enough(&cuda[0].clone().into_tensor(), Approximation::Approximate)?;
        Ok(())
    }
}

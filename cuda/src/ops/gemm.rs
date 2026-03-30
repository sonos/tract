use crate::context::StreamExt;
use crate::kernels::matmul::GgmlGemm;
use crate::ops::GgmlQuantQ81Fact;
use crate::utils::get_ggml_q81_fact;

use anyhow::{bail, ensure};
use derive_new::new;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::as_quant_fact;

#[derive(Debug, new, Default, Clone, PartialEq, Eq)]
pub struct CudaGgmlGemm;

impl Op for CudaGgmlGemm {
    fn name(&self) -> StaticName {
        "CudaGgmlGemm".into()
    }

    op_as_typed_op!();
}

impl CudaGgmlGemm {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [a, b] = inputs else {
            bail!("Expects 2 inputs");
        };

        if a.datum_type.is_number() && b.datum_type.is_number() && a.is_plain() && b.is_plain() {
            ensure!(a.rank() == b.rank());
            ensure!(a.rank() >= 2);
            ensure!(a.shape[a.rank() - 1] == b.shape[b.rank() - 1]);
            let out_shape = GgmlGemm.output_shape(&a.shape, &b.shape);
            Ok(tvec![a.datum_type().unwrap().fact(out_shape)])
        } else if as_quant_fact(inputs[1], &Q4_0).is_some() {
            let Some(b_ggml_qf) =
                inputs[0].exotic_fact.as_ref().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>())
            else {
                bail!("Expected GGML Q81 activations for Q40 MM")
            };
            let a_shape: ShapeFact =
                a.shape.iter().cloned().chain(b_ggml_qf.in_shape().to_owned()).collect();
            // b.shape already carries the full logical dimensions after the
            // exotic-storage refactoring — no need to chain with BlockQuantFact.shape().
            let out_shape = GgmlGemm.output_shape(&a_shape, &b.shape);
            Ok(tvec![DatumType::F32.fact(out_shape)])
        } else {
            bail!("Unsupported datum type configuration for GEMM")
        }
    }
}
impl EvalOp for CudaGgmlGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (act_raw, weights_raw) = args_2!(inputs);
        let activs = act_raw
            .to_device_tensor()
            .with_context(|| format!("A tensor is not a cuda tensor: {act_raw:?}"))?;
        let weights = weights_raw
            .to_device_tensor()
            .with_context(|| format!("B tensor is not a cuda tensor {weights_raw:?}"))?;

        let (activ_shape, weights_shape) =
            crate::kernels::matmul::get_concrete_shapes(activs, weights)?;

        let out_shape = GgmlGemm.output_shape(&activ_shape, &weights_shape);
        let out_dt =
            if get_ggml_q81_fact(activs).is_some() { DatumType::F32 } else { activs.datum_type() };
        let out =
            tract_gpu::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;

        tract_gpu::with_stream(|stream| {
            let stream = stream.cuda()?;
            GgmlGemm.dispatch_eval(stream, activs, weights, &out)
        })?;

        Ok(tvec![out.into_tensor().into_tvalue()])
    }
}

impl TypedOp for CudaGgmlGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |input_facts| {
            self.resolve_output_facts(input_facts)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| {
            let fma = self.resolve_output_facts(input_facts)?[0].shape.iter().product::<TDim>()
                * input_facts[0].shape.last().unwrap();
            if input_facts[0].datum_type == f16::datum_type() {
                Ok(tvec!((Cost::FMA(f16::datum_type()), fma)))
            } else {
                Ok(tvec!((Cost::FMA(f32::datum_type()), fma)))
            }
        })
        .with_context(|| format!("Error while computing cost for {:?}", self.name()))
    }

    as_op!();
}

use crate::context::CUDA_STREAM;
use crate::kernels::Matmul;
use crate::utils::get_q40_fact;

use anyhow::{bail, ensure};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::as_q40_fact;

#[derive(Debug, Default, Clone)]
pub struct CudaGemm;

impl Op for CudaGemm {
    fn name(&self) -> StaticName {
        "CudaGemm".into()
    }

    op_as_typed_op!();
}

impl CudaGemm {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [a, b] = inputs else {
            bail!("Expects 2 inputs");
        };

        if a.datum_type.is_number() && b.datum_type.is_number() {
            ensure!(a.rank() == b.rank());
            ensure!(a.rank() >= 2);
            ensure!(a.shape[a.rank() - 1] == b.shape[b.rank() - 1]);
            let out_shape = Matmul::output_shape(&a.shape, &b.shape);
            Ok(tvec![a.datum_type().unwrap().fact(out_shape)])
        } else if let Some(opf) = as_q40_fact(inputs[1]) {
            let b_shape: ShapeFact =
                b.shape.iter().cloned().chain(opf.shape().iter().map(|d| d.to_dim())).collect();
            let out_shape = Matmul::output_shape(&a.shape, &b_shape);
            Ok(tvec![a.datum_type().unwrap().fact(out_shape)])
        } else {
            bail!("Unsupported datum type configuration for GEMM")
        }
    }
}
impl EvalOp for CudaGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (a_opaque, b_opaque) = args_2!(inputs);
        let a = a_opaque
            .to_device_tensor()
            .with_context(|| format!("A tensor is not a metal tensor: {a_opaque:?}"))?;
        let b = b_opaque
            .to_device_tensor()
            .with_context(|| format!("B tensor is not a metal tensor {b_opaque:?}"))?;

        ensure!((a.datum_type() == b.datum_type()) || get_q40_fact(b).is_some());

        let b_shape = get_q40_fact(b)
            .map(|bqf| b.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        let c_dt = a.datum_type();
        let c_shape = Matmul::output_shape(a.shape(), &b_shape);
        let c = tract_gpu::session_handler::make_tensor_for_node(session, node_id, c_dt, &c_shape)?;

        CUDA_STREAM.with(|stream| Matmul.dispatch_eval(stream.clone(), a, b, &c))?;

        Ok(tvec![c.into_opaque_tensor().into_tvalue()])
    }
}

impl TypedOp for CudaGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |input_facts| {
            self.resolve_output_facts(input_facts)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| {
            let fma = Matmul::output_shape(&input_facts[0].shape, &input_facts[1].shape)
                .iter()
                .product::<TDim>()
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

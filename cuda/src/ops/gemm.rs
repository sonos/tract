use crate::context::CUDA_STREAM;
use crate::kernels::matmul::{GemmImpl, GemmKernel};
use crate::ops::GgmlQuantQ81Fact;
use crate::utils::{get_ggml_q81_fact, get_quant_fact};

use anyhow::{bail, ensure};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::tensor::DeviceTensorExt;
use tract_gpu::utils::as_quant_fact;

#[derive(Debug, Default, Clone)]
pub struct CudaGemm<K: GemmKernel> {
    pub kernel: GemmImpl<K>,
}

impl<K: GemmKernel> CudaGemm<K> {
    pub fn new(transpose_a: bool, transpose_b: bool) -> Self {
        Self { kernel: GemmImpl::<K>::new(transpose_a, transpose_b) }
    }
}

impl<K: GemmKernel + 'static> Op for CudaGemm<K> {
    fn name(&self) -> StaticName {
        "CudaGemm".into()
    }

    op_as_typed_op!();
}

impl<K: GemmKernel> CudaGemm<K> {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [a, b] = inputs else {
            bail!("Expects 2 inputs");
        };

        if a.datum_type.is_number() && b.datum_type.is_number() {
            ensure!(a.rank() == b.rank());
            ensure!(a.rank() >= 2);
            ensure!(a.shape[a.rank() - 1] == b.shape[b.rank() - 1]);
            let out_shape = self.kernel.output_shape(&a.shape, &b.shape);
            Ok(tvec![a.datum_type().unwrap().fact(out_shape)])
        } else if let Some(a_bqf) = as_quant_fact(inputs[1], &Q4_0) {
            let Some(b_ggml_qf) = inputs[0].opaque_fact.as_ref().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>())
            else { 
                bail!("Expected GGML Q81 activations for Q40 MM")
            };
            let a_shape: ShapeFact = 
                a.shape.iter().cloned().chain(b_ggml_qf.in_shape()).collect();
            let b_shape: ShapeFact =
                b.shape.iter().cloned().chain(a_bqf.shape().iter().map(|d| d.to_dim())).collect();
            let out_shape = self.kernel.output_shape(&a_shape, &b_shape);
            Ok(tvec![DatumType::F32.fact(out_shape)])
        } else {
            bail!("Unsupported datum type configuration for GEMM")
        }
    }
}
impl<K: GemmKernel> EvalOp for CudaGemm<K> {
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
            .with_context(|| format!("A tensor is not a cuda tensor: {a_opaque:?}"))?;
        let b = b_opaque
            .to_device_tensor()
            .with_context(|| format!("B tensor is not a cuda tensor {b_opaque:?}"))?;

        ensure!((a.datum_type() == b.datum_type()));

        let a_shape = get_ggml_q81_fact(a)
            .map(|bqf| a.shape().iter().cloned().chain(bqf.in_shape().iter().map(|d| d.as_i64().unwrap() as usize)).collect())
            .unwrap_or(a.shape().to_vec());
        let b_shape = get_quant_fact(b, &Q4_0)
            .map(|bqf| b.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        let c_shape = self.kernel.output_shape(&a_shape, &b_shape);
        let c_dt = if get_ggml_q81_fact(a).is_some() { DatumType::F32 } else { a.datum_type() };
        let c = tract_gpu::session_handler::make_tensor_for_node(session, node_id, c_dt, &c_shape)?;

        CUDA_STREAM.with(|stream| self.kernel.dispatch_eval(stream, a, b, &c))?;

        Ok(tvec![c.into_opaque_tensor().into_tvalue()])
    }
}

impl<K: GemmKernel + 'static> TypedOp for CudaGemm<K> {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |input_facts| {
            self.resolve_output_facts(input_facts)
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        tract_gpu::utils::get_device_facts(inputs, |input_facts| {
            let fma = self.resolve_output_facts(inputs)?[0].shape.iter().product::<TDim>()
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

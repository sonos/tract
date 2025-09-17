use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q8_1};
use tract_gpu::session_handler::make_scalar_opaque_tensor_for_node;
use tract_gpu::tensor::DeviceTensorExt;

use crate::context::CUDA_STREAM;
use crate::kernels::matmul::GgmlQuantQ81;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GgmlQuantQ81Fact {
    in_shape: TVec<TDim>,
    out_shape: TVec<TDim>,
}

impl GgmlQuantQ81Fact {
    pub fn in_shape(&self) -> TVec<TDim> {
        self.in_shape.clone()
    }

    pub fn out_shape(&self) -> TVec<TDim> {
        self.out_shape.clone()
    }
}

impl OpaqueFact for GgmlQuantQ81Fact {
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        let Some(other) = other.downcast_ref::<Self>() else { return false };
        (other.in_shape == self.in_shape) && (other.out_shape == other.out_shape)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(self.out_shape.iter().product::<TDim>() * Q8_1.block_bytes() / Q8_1.block_len())
    }
}

#[derive(Clone, Debug, Hash)]
pub struct CudaGgmlQuantQ81 {
    out_fact: GgmlQuantQ81Fact,
}

impl CudaGgmlQuantQ81 {
    pub fn new(in_shape: &[TDim]) -> TractResult<Self> {
        let in_shape = in_shape.into();
        let out_shape = GgmlQuantQ81::output_shape(&in_shape)?;

        let out_fact = GgmlQuantQ81Fact { in_shape, out_shape };

        Ok(Self { out_fact })
    }
}
impl Op for CudaGgmlQuantQ81 {
    fn name(&self) -> StaticName {
        "CudaGgmlQuantQ81Op".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaGgmlQuantQ81 {
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            let opaque = args_1!(inputs);
            let input = opaque.to_device_tensor()?;

            let resolved_out_fact = 
            GgmlQuantQ81Fact {
                in_shape: self.out_fact
                    .in_shape()
                    .iter()
                    .map(|d| d.eval(&session.resolved_symbols))
                    .collect(),
                out_shape: self.out_fact
                    .out_shape()
                    .iter()
                    .map(|d| d.eval(&session.resolved_symbols))
                    .collect(),
            };
            let output =
                make_scalar_opaque_tensor_for_node(session, node_id, Box::new(resolved_out_fact))?;

            GgmlQuantQ81.dispatch_eval(stream, input, &output)?;

            Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
        })
    }

    fn is_stateless(&self) -> bool {
        true
    }
}

impl TypedOp for CudaGgmlQuantQ81 {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        tract_gpu::utils::facts_to_device_facts(inputs, |_| {
            let fact =
                TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(self.out_fact.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

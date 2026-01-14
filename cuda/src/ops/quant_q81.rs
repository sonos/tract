use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q8_1};
use tract_gpu::session_handler::make_scalar_opaque_tensor_for_node;
use tract_gpu::tensor::DeviceTensorExt;

use crate::context::CUDA_STREAM;
use crate::kernels::matmul::quant_act_q81::GgmlQuantQ81;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GgmlQuantQ81Fact {
    pub in_fact: ShapeFact,
    pub out_fact: ShapeFact,
}

impl GgmlQuantQ81Fact {
    pub fn in_shape(&self) -> &[TDim] {
        self.in_fact.dims()
    }

    pub fn out_shape(&self) -> &[TDim] {
        self.out_fact.dims()
    }

    pub fn concrete_in_shape(&self) -> TractResult<&[usize]> {
        self.in_fact.as_concrete().context("Expected concrete shape")
    }

    pub fn concrete_out_shape(&self) -> TractResult<&[usize]> {
        self.out_fact.as_concrete().context("Expected concrete shape")
    }

    pub fn eval(&self, values: &SymbolValues) -> TractResult<Self> {
        Ok(Self {
            in_fact: self.in_fact.eval(values)?.into_owned(),
            out_fact: self.out_fact.eval(values)?.into_owned(),
        })
    }
}

impl OpaqueFact for GgmlQuantQ81Fact {
    fn same_as(&self, other: &dyn OpaqueFact) -> bool {
        let Some(other) = other.downcast_ref::<Self>() else { return false };
        (other.in_fact == self.in_fact) && (other.out_fact == self.out_fact)
    }

    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(self.out_fact.iter().product::<TDim>() * Q8_1.block_bytes() / Q8_1.block_len())
    }
}

#[derive(Clone, Debug, Hash)]
pub struct CudaGgmlQuantQ81 {
    io_facts: GgmlQuantQ81Fact,
}

impl CudaGgmlQuantQ81 {
    pub fn new(in_fact: ShapeFact) -> TractResult<Self> {
        let out_fact = GgmlQuantQ81::output_shape_fact(&in_fact)?;
        let io_facts = GgmlQuantQ81Fact { in_fact, out_fact };
        Ok(Self { io_facts })
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

            let resolved_io_facts = self.io_facts.eval(&session.resolved_symbols)?;

            let output =
                make_scalar_opaque_tensor_for_node(session, node_id, Box::new(resolved_io_facts))?;

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
                TypedFact::dt_scalar(DatumType::Opaque).with_opaque_fact(self.io_facts.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = Self::new(self.io_facts.in_fact.eval(values)?.into_owned())?;
        target.wire_node(&node.name, op, &[mapping[&node.inputs[0]]])
    }
    as_op!();
}

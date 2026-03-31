use crate::context::StreamExt;
pub use crate::kernels::BinOps;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalBinOp(pub BinOps);

impl MetalBinOp {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (a, b) = (inputs[0], inputs[1]);
        if a.rank() != b.rank() {
            bail!(
                "Typed ops require rank match. Invalid inputs for {}: {{a: {:?}, b: {:?}}}",
                self.name(),
                a.shape,
                b.shape
            );
        }

        let out_shape = self.0.output_shape(&a.shape, &b.shape)?;
        let out_dt = self.0.output_datum_type(a.datum_type, b.datum_type)?;
        Ok(tvec!(out_dt.fact(out_shape)))
    }
}

impl Op for MetalBinOp {
    fn name(&self) -> StaticName {
        format!("Metal{}", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (a_raw, b_raw) = args_2!(inputs);
        let a = a_raw.to_device_tensor()?;
        let b = b_raw.to_device_tensor()?;
        let out_shape = self.0.output_shape(a.shape(), b.shape())?;
        let out_dt = self.0.output_datum_type(a.datum_type(), b.datum_type())?;
        let output =
            tract_gpu::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;
        tract_gpu::with_stream(|stream| {
            let stream = stream.metal()?;
            self.0
                .dispatch_eval(stream, a, b, &output)
                .with_context(|| "Error while dispatching eval for Metal Bin Op")
        })?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.resolve_output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

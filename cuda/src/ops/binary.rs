use crate::context::CUDA_STREAM;
use crate::kernels::BinOps;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone)]
pub struct CudaBinOp(pub BinOps);

impl CudaBinOp {
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

impl Op for CudaBinOp {
    fn name(&self) -> StaticName {
        format!("Cuda{}", self.0.name()).into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<CudaBinOp>() else { return false };
        self.0 == other.0
    }

    op_as_typed_op!();
}

impl EvalOp for CudaBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (opaque_a, opaque_b) = args_2!(inputs);

        let a = opaque_a.to_device_tensor()?;
        let b = opaque_b.to_device_tensor()?;
        let out_shape = self.0.output_shape(a.shape(), b.shape())?;
        let out_dt = self.0.output_datum_type(a.datum_type(), b.datum_type())?;
        let output =
            tract_gpu::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;

        CUDA_STREAM.with(|stream| {
            self.0
                .dispatch_eval(stream, a, b, &output)
                .with_context(|| "Error while dispatching eval for Metal Bin Op")
        })?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for CudaBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.resolve_output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

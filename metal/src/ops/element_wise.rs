pub use crate::kernels::ElementWiseOps;
use crate::utils::with_borrowed_metal_stream;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone)]
pub struct MetalElementWiseOp(pub ElementWiseOps);

impl Op for MetalElementWiseOp {
    fn name(&self) -> Cow<str> {
        format!("Metal{}", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<MetalElementWiseOp>() else { return false };
        self.0 == other.0
    }

    op_as_typed_op!();
}

impl EvalOp for MetalElementWiseOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        with_borrowed_metal_stream(|stream| {
            let opaque_a = args_1!(inputs);
            let a = opaque_a.to_device_tensor()?;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                a.datum_type(),
                a.shape(),
            )?;
            self.0.dispatch_eval(stream, a, &output)?;
            Ok(tvec![output.into_opaque_tensor().into_tvalue()])
        })
    }
}

impl TypedOp for MetalElementWiseOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| Ok(tvec!(facts[0].without_value())))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

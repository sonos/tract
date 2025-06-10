use crate::kernels;
use crate::utils::with_borrowed_metal_stream;
use derive_new::new;
use std::fmt::Debug;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, new, Hash)]
pub struct MetalMultiBroadcastTo {
    pub shape: ShapeFact,
}

impl Op for MetalMultiBroadcastTo {
    fn name(&self) -> StaticName {
        "MetalMultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MetalMultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let shape = self.shape.eval_to_usize(&session.resolved_symbols)?;
        let input = opaque.to_device_tensor()?;
        let output = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &shape,
        )?;

        with_borrowed_metal_stream(|stream| {
            kernels::array::MultiBroadcast.dispatch_eval(stream, input, 0, &output)
        })?;
        Ok(tvec![output.into_opaque_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalMultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].datum_type.fact(self.shape.clone());
            fact.uniform.clone_from(&inputs[0].uniform);
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

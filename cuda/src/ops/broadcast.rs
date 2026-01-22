use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::context::CUDA_STREAM;
use crate::kernels;

#[derive(Debug, Clone, new, Hash)]
pub struct CudaMultiBroadcastTo {
    pub shape: ShapeFact,
}

impl Op for CudaMultiBroadcastTo {
    fn name(&self) -> StaticName {
        "CudaMultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

impl EvalOp for CudaMultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
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

        CUDA_STREAM.with(|stream| {
            kernels::array::MultiBroadcast.dispatch_eval(stream, input, 0, &output)
        })?;
        Ok(tvec![output.into_opaque_tensor().into_tvalue()])
    }
}

impl TypedOp for CudaMultiBroadcastTo {
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

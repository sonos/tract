use crate::kernels::array::RotateHalf;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use derive_new::new;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalRotateHalf;

impl Op for MetalRotateHalf {
    fn name(&self) -> Cow<str> {
        "MetalRotateHalf".into()
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalRotateHalf);

impl MetalEvalOp for MetalRotateHalf {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_device_tensor()?;
        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), input.shape())?;
        RotateHalf.dispatch_eval(stream, input, &output)?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalRotateHalf {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

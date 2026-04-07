use crate::tensor::DeviceTensorExt;
use crate::utils::compute_broadcast_strides;
use tract_core::internal::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuMultiBroadcastTo {
    pub shape: ShapeFact,
}

impl GpuMultiBroadcastTo {
    pub fn new(shape: ShapeFact) -> Self {
        Self { shape }
    }
}

impl Op for GpuMultiBroadcastTo {
    fn name(&self) -> StaticName {
        "GpuMultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuMultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let shape = self.shape.eval_to_usize(&session.resolved_symbols)?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &shape,
        )?;

        // Pad input shape/strides to output rank for broadcasting
        let mut input_strides = vec![input.strides()[0]; output.rank() - input.rank()];
        input_strides.extend(input.strides());
        let mut input_shape = vec![1usize; output.rank() - input.rank()];
        input_shape.extend(input.shape());
        let broadcast_strides: TVec<isize> =
            compute_broadcast_strides(&input_shape, &input_strides)?;

        let ctx = crate::device::get_context()?;
        ctx.copy_nd(input, 0, &broadcast_strides, &output, 0, output.shape(), output.strides())?;
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuMultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let mut fact = facts[0].datum_type.fact(self.shape.clone());
            fact.uniform.clone_from(&inputs[0].uniform);
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

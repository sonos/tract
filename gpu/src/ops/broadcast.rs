use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

pub type DispatchBroadcastFn = fn(&DeviceTensor, usize, &DeviceTensor) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuMultiBroadcastTo {
    pub shape: ShapeFact,
    pub backend_name: &'static str,
    pub dispatch: DispatchBroadcastFn,
}

impl std::fmt::Debug for GpuMultiBroadcastTo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}MultiBroadcastTo({:?})", self.backend_name, self.shape)
    }
}

impl PartialEq for GpuMultiBroadcastTo {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.shape == other.shape
    }
}

impl Eq for GpuMultiBroadcastTo {}

impl std::hash::Hash for GpuMultiBroadcastTo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.shape.hash(state);
    }
}

impl Op for GpuMultiBroadcastTo {
    fn name(&self) -> StaticName {
        format!("{}MultiBroadcastTo", self.backend_name).into()
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

        (self.dispatch)(input, 0, &output)?;
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

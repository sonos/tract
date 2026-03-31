use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseMiniOp;

pub type DispatchElementWiseFn =
    fn(&dyn ElementWiseMiniOp, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone)]
pub struct GpuElementWise {
    pub backend_name: &'static str,
    pub mini_op: Box<dyn ElementWiseMiniOp>,
    pub dispatch: DispatchElementWiseFn,
}

impl std::fmt::Debug for GpuElementWise {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "GpuElementWise({}{:?})", self.backend_name, self.mini_op)
    }
}

impl PartialEq for GpuElementWise {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.mini_op == other.mini_op
    }
}

impl Eq for GpuElementWise {}

impl std::hash::Hash for GpuElementWise {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.mini_op.name().hash(state);
    }
}

impl Op for GpuElementWise {
    fn name(&self) -> StaticName {
        format!("{}{}", self.backend_name, self.mini_op.name()).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuElementWise {
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
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            input.shape(),
        )?;
        (self.dispatch)(&*self.mini_op, input, &output)
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuElementWise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

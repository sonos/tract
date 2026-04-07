use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

pub type DispatchGeluApproximateFn = fn(bool, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuGeluApproximate {
    pub fast_impl: bool,
    pub backend_name: &'static str,
    pub dispatch: DispatchGeluApproximateFn,
}

impl std::fmt::Debug for GpuGeluApproximate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}GeluApproximate(fast_impl: {})", self.backend_name, self.fast_impl)
    }
}

impl PartialEq for GpuGeluApproximate {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.fast_impl == other.fast_impl
    }
}

impl Eq for GpuGeluApproximate {}

impl std::hash::Hash for GpuGeluApproximate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.fast_impl.hash(state);
    }
}

impl Op for GpuGeluApproximate {
    fn name(&self) -> StaticName {
        format!("{}GeluApproximate", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuGeluApproximate {
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
        (self.dispatch)(self.fast_impl, input, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuGeluApproximate {
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

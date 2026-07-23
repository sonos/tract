use crate::session_handler::make_tensor_for_node;
use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;

pub type DispatchCausalConv1dUpdateFn = fn(
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
) -> TractResult<()>;

#[derive(Clone, Debug)]
pub struct GpuCausalConv1dUpdate {
    pub backend_name: &'static str,
    pub dispatch: DispatchCausalConv1dUpdateFn,
}

impl PartialEq for GpuCausalConv1dUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
    }
}
impl Eq for GpuCausalConv1dUpdate {}
impl std::hash::Hash for GpuCausalConv1dUpdate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
    }
}

impl Op for GpuCausalConv1dUpdate {
    fn name(&self) -> StaticName {
        format!("{}CausalConv1dUpdate", self.backend_name).into()
    }
    op_as_typed_op!();
}

impl EvalOp for GpuCausalConv1dUpdate {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (input, weight, state) = args_3!(inputs);
        let input = input.to_device_tensor()?;
        let weight = weight.to_device_tensor()?;
        let state = state.to_device_tensor()?;
        let output = make_tensor_for_node(session, node_id, DatumType::F16, input.shape())?;
        let final_state = DeviceTensor::uninitialized_dt(DatumType::F16, state.shape())?;
        (self.dispatch)(input, weight, state, &output, &final_state)?;
        Ok(tvec![output.into_tensor().into_tvalue(), final_state.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuCausalConv1dUpdate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 3);
            ensure!(facts.iter().all(|fact| fact.datum_type == DatumType::F16));
            Ok(tvec![facts[0].without_value(), facts[2].without_value()])
        })
        .with_context(|| format!("invalid facts for {}", self.name()))
    }
    as_op!();
}

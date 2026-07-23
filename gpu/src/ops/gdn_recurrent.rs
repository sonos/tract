use crate::session_handler::make_tensor_for_node;
use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;

pub type DispatchGdnRecurrentFn = fn(
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
) -> TractResult<()>;

#[derive(Clone, Debug)]
pub struct GpuGatedDeltaNetRecurrent {
    pub backend_name: &'static str,
    pub dispatch: DispatchGdnRecurrentFn,
}

impl PartialEq for GpuGatedDeltaNetRecurrent {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
    }
}
impl Eq for GpuGatedDeltaNetRecurrent {}
impl std::hash::Hash for GpuGatedDeltaNetRecurrent {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
    }
}

impl Op for GpuGatedDeltaNetRecurrent {
    fn name(&self) -> StaticName {
        format!("{}GatedDeltaNetRecurrent", self.backend_name).into()
    }
    op_as_typed_op!();
}

impl EvalOp for GpuGatedDeltaNetRecurrent {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6);
        let tensors = inputs
            .iter()
            .map(|value| value.to_device_tensor())
            .collect::<TractResult<TVec<_>>>()?;
        let output = make_tensor_for_node(session, node_id, DatumType::F16, tensors[0].shape())?;
        // The memory arena is keyed by node, so a second output cannot use the
        // same arena slot as the first one.
        let final_state = DeviceTensor::uninitialized_dt(DatumType::F32, tensors[5].shape())?;
        (self.dispatch)(
            tensors[0],
            tensors[1],
            tensors[2],
            tensors[3],
            tensors[4],
            tensors[5],
            &output,
            &final_state,
        )?;
        Ok(tvec![output.into_tensor().into_tvalue(), final_state.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuGatedDeltaNetRecurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 6);
            ensure!(facts[0].datum_type == DatumType::F16);
            ensure!(facts[1].datum_type == DatumType::F16);
            ensure!(facts[2].datum_type == DatumType::F16);
            ensure!(facts[3].datum_type == DatumType::F32);
            ensure!(facts[4].datum_type == DatumType::F16);
            ensure!(facts[5].datum_type == DatumType::F32);
            Ok(tvec![facts[0].without_value(), facts[5].without_value()])
        })
        .with_context(|| format!("invalid facts for {}", self.name()))
    }
    as_op!();
}

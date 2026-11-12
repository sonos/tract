use crate::tensor::{DeviceTensor, DeviceTensorExt};
use crate::turn_handler::make_tensor_for_node_output;
use tract_core::internal::*;

/// Runs the convolution: input, weight, initial state and an optional bias in, the output
/// and the final state written.
pub type DispatchCausalConv1dUpdateFn = fn(
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    Option<&DeviceTensor>,
    &DeviceTensor,
    &DeviceTensor,
) -> TractResult<()>;

/// Backend-agnostic `tract_transformers` `CausalConv1dUpdate` for f16 tensors: input,
/// weight, initial state and an optional bias, as the CPU op takes them.
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
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(matches!(inputs.len(), 3 | 4));
        let t = inputs.iter().map(|v| v.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output = make_tensor_for_node_output(ctx, 0, DatumType::F16, t[0].shape())?;
        let final_state = make_tensor_for_node_output(ctx, 1, DatumType::F16, t[2].shape())?;
        (self.dispatch)(t[0], t[1], t[2], t.get(3).copied(), &output, &final_state)?;
        Ok(tvec![output.into_tensor().into_tvalue(), final_state.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuCausalConv1dUpdate {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(matches!(facts.len(), 3 | 4));
            ensure!(facts.iter().all(|fact| fact.datum_type == DatumType::F16));
            Ok(tvec![facts[0].without_value(), facts[2].without_value()])
        })
        .with_context(|| format!("invalid facts for {}", self.name()))
    }
    as_op!();
}

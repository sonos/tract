use crate::tensor::DeviceTensor;
use crate::tensor::DeviceTensorExt;
use crate::turn_handler::make_tensor_for_node_output;
use tract_core::internal::*;

/// Runs the recurrence: x, b, c, log_decay and the initial state in, the output and the
/// final state written.
pub type DispatchMamba2RecurrentFn = fn(
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
    &DeviceTensor,
) -> TractResult<()>;

/// Backend-agnostic `tract_transformers` `Mamba2Recurrent`: same inputs, layout and outputs
/// (x `[b, S, H, P]`, b/c `[b, S, G, N]`, log_decay `[b, S, H]`, state `[b, H, P, N]`; the
/// output in x's datum type, the final state in the initial state's).
#[derive(Clone, Debug)]
pub struct GpuMamba2Recurrent {
    pub backend_name: &'static str,
    pub dispatch: DispatchMamba2RecurrentFn,
}

impl PartialEq for GpuMamba2Recurrent {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name
    }
}

impl Eq for GpuMamba2Recurrent {}

impl std::hash::Hash for GpuMamba2Recurrent {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
    }
}

impl Op for GpuMamba2Recurrent {
    fn name(&self) -> StaticName {
        format!("{}Mamba2Recurrent", self.backend_name).into()
    }
    op_as_typed_op!();
}

impl EvalOp for GpuMamba2Recurrent {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 5);
        let t = inputs.iter().map(|v| v.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let output = make_tensor_for_node_output(ctx, 0, t[0].datum_type(), t[0].shape())?;
        let final_state = make_tensor_for_node_output(ctx, 1, t[4].datum_type(), t[4].shape())?;
        (self.dispatch)(t[0], t[1], t[2], t[3], t[4], &output, &final_state)
            .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        Ok(tvec![output.into_tensor().into_tvalue(), final_state.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuMamba2Recurrent {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 5);
            Ok(tvec![facts[0].without_value(), facts[4].without_value()])
        })
        .with_context(|| format!("Error while computing facts for {}", self.name()))
    }
    as_op!();
}

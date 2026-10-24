use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

pub type DispatchNanReplaceFn = fn(f32, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

/// Replaces every NaN of its input with `value`: core's `Iff(IsNan(x), value, x)` as one
/// elementwise op, for backends with no boolean tensors to carry the condition.
#[derive(Clone, new)]
pub struct GpuNanReplace {
    pub value: f32,
    pub backend_name: &'static str,
    pub dispatch: DispatchNanReplaceFn,
}

impl std::fmt::Debug for GpuNanReplace {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}NanReplace(value: {})", self.backend_name, self.value)
    }
}

impl PartialEq for GpuNanReplace {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.value.to_bits() == other.value.to_bits()
    }
}

impl Eq for GpuNanReplace {}

impl std::hash::Hash for GpuNanReplace {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.value.to_bits().hash(state);
    }
}

impl Op for GpuNanReplace {
    fn name(&self) -> StaticName {
        format!("{}NanReplace", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuNanReplace {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let output =
            crate::turn_handler::make_tensor_for_node(ctx, input.datum_type(), input.shape())?;
        (self.dispatch)(self.value, input, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuNanReplace {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(facts[0].datum_type.fact(facts[0].shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

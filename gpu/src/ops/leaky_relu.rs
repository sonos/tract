use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;

use crate::tensor::DeviceTensor;

pub type DispatchLeakyReluFn = fn(f32, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone)]
pub struct GpuLeakyRelu {
    pub alpha: f32,
    pub backend_name: &'static str,
    pub dispatch: DispatchLeakyReluFn,
}

impl GpuLeakyRelu {
    pub fn new(alpha: f32, backend_name: &'static str, dispatch: DispatchLeakyReluFn) -> Self {
        Self { alpha, backend_name, dispatch }
    }
}

impl std::fmt::Debug for GpuLeakyRelu {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}LeakyRelu(alpha: {})", self.backend_name, self.alpha)
    }
}

impl PartialEq for GpuLeakyRelu {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.alpha == other.alpha
    }
}

impl Eq for GpuLeakyRelu {}

impl std::hash::Hash for GpuLeakyRelu {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.alpha.to_bits().hash(state);
    }
}

impl Op for GpuLeakyRelu {
    fn name(&self) -> StaticName {
        format!("{}LeakyRelu", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuLeakyRelu {
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
        (self.dispatch)(self.alpha, input, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuLeakyRelu {
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

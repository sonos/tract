use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use std::sync::Arc;
use tract_core::internal::*;

pub type DispatchRmsNormFn = fn(&DeviceTensor, usize, &Tensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
    pub backend_name: &'static str,
    pub dispatch: DispatchRmsNormFn,
}

impl std::fmt::Debug for GpuRmsNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}RmsNorm(axis: {:?}, eps: {:?})", self.backend_name, self.axis, self.eps)
    }
}

impl PartialEq for GpuRmsNorm {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.axis == other.axis && self.eps == other.eps
    }
}

impl Eq for GpuRmsNorm {}

impl std::hash::Hash for GpuRmsNorm {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.axis.hash(state);
        self.eps.hash(state);
    }
}

impl Op for GpuRmsNorm {
    fn name(&self) -> StaticName {
        format!("{}RmsNorm", self.backend_name).into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuRmsNorm {
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
        (self.dispatch)(input, self.axis, &self.eps, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuRmsNorm {
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

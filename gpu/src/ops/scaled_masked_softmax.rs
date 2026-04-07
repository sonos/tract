use crate::tensor::{DeviceTensor, DeviceTensorExt};
use derive_new::new;
use tract_core::internal::*;

/// A = SOFTMAX(INPUT * SCALE + MASK, AXIS=2)
/// Only input of rank of 3 is supported
pub type DispatchScaledMaskedSoftmaxFn =
    fn(&DeviceTensor, &Tensor, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone, new)]
pub struct GpuScaledMaskedSoftmax {
    pub scale: Arc<Tensor>,
    pub backend_name: &'static str,
    pub dispatch: DispatchScaledMaskedSoftmaxFn,
}

impl std::fmt::Debug for GpuScaledMaskedSoftmax {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}ScaledMaskedSoftmax", self.backend_name)
    }
}

impl PartialEq for GpuScaledMaskedSoftmax {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.scale == other.scale
    }
}
impl Eq for GpuScaledMaskedSoftmax {}

impl std::hash::Hash for GpuScaledMaskedSoftmax {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.scale.hash(state);
    }
}

impl Op for GpuScaledMaskedSoftmax {
    fn name(&self) -> StaticName {
        format!("{}ScaledMaskedSoftmax", self.backend_name).into()
    }
    op_as_typed_op!();
}

impl EvalOp for GpuScaledMaskedSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (input_val, mask_val) = args_2!(inputs);
        let input = input_val.to_device_tensor()?;
        let mask = mask_val.to_device_tensor()?;
        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            input.shape(),
        )?;
        (self.dispatch)(input, &self.scale, mask, &output)?;
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuScaledMaskedSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            ensure!(facts.len() == 2);
            let dt = facts[0].datum_type;
            ensure!(dt == facts[1].datum_type);
            ensure!(facts[0].rank() <= 5);
            ensure!(facts[0].rank() >= 2);
            ensure!(facts[0].rank() == facts[1].rank());
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }
    as_op!();
}

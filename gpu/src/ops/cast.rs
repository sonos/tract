use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;

use crate::tensor::DeviceTensor;

pub type DispatchCastFn = fn(&DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone)]
pub struct GpuCast {
    pub to: DatumType,
    pub backend_name: &'static str,
    pub dispatch: DispatchCastFn,
    pub is_supported_dt: fn(DatumType) -> bool,
}

impl GpuCast {
    pub fn new(
        to: DatumType,
        backend_name: &'static str,
        dispatch: DispatchCastFn,
        is_supported_dt: fn(DatumType) -> bool,
    ) -> Option<Self> {
        is_supported_dt(to).then_some(Self { to, backend_name, dispatch, is_supported_dt })
    }

    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        (self.is_supported_dt)(dt)
    }
}

impl std::fmt::Debug for GpuCast {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Cast({:?})", self.backend_name, self.to)
    }
}

impl PartialEq for GpuCast {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.to == other.to
    }
}

impl Eq for GpuCast {}

impl std::hash::Hash for GpuCast {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.to.hash(state);
    }
}

impl Op for GpuCast {
    fn name(&self) -> StaticName {
        format!("{}Cast", self.backend_name).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuCast {
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
        if input.datum_type() == self.to {
            Ok(tvec!(input_value))
        } else {
            let output = crate::session_handler::make_tensor_for_node(
                session,
                node_id,
                self.to,
                input.shape(),
            )?;
            (self.dispatch)(input, &output)?;
            Ok(tvec![output.into_tensor().into_tvalue()])
        }
    }
}

impl TypedOp for GpuCast {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(self.to.fact(facts[0].shape.clone())))
        })
        .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

use crate::tensor::{DeviceTensor, DeviceTensorExt};
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;

pub type DispatchBinOpFn =
    fn(&dyn BinMiniOp, &DeviceTensor, &DeviceTensor, &DeviceTensor) -> TractResult<()>;

#[derive(Clone)]
pub struct GpuBinOp {
    pub backend_name: &'static str,
    pub mini_op: Box<dyn BinMiniOp>,
    pub dispatch: DispatchBinOpFn,
}

impl std::fmt::Debug for GpuBinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "GpuBinOp({}{:?})", self.backend_name, self.mini_op)
    }
}

impl PartialEq for GpuBinOp {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.mini_op == other.mini_op
    }
}

impl Eq for GpuBinOp {}

impl std::hash::Hash for GpuBinOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.mini_op.name().hash(state);
    }
}

impl GpuBinOp {
    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (a, b) = (inputs[0], inputs[1]);
        if a.rank() != b.rank() {
            bail!(
                "Typed ops require rank match. Invalid inputs for {}: {{a: {:?}, b: {:?}}}",
                self.name(),
                a.shape,
                b.shape
            );
        }
        let out_shape = tract_core::broadcast::multi_broadcast(&[&a.shape, &b.shape])
            .with_context(|| format!("Error while broadcasting {:?} {:?}", a.shape, b.shape))?;
        let out_dt = self.mini_op.result_datum_type(a.datum_type, b.datum_type)?;
        Ok(tvec!(out_dt.fact(out_shape)))
    }
}

impl Op for GpuBinOp {
    fn name(&self) -> StaticName {
        format!("{}{}", self.backend_name, self.mini_op.name()).into()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuBinOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (a_val, b_val) = args_2!(inputs);
        let a = a_val.to_device_tensor()?;
        let b = b_val.to_device_tensor()?;
        let out_shape = tract_core::broadcast::multi_broadcast(&[a.shape(), b.shape()])?;
        let out_dt = self.mini_op.result_datum_type(a.datum_type(), b.datum_type())?;
        let output =
            crate::session_handler::make_tensor_for_node(session, node_id, out_dt, &out_shape)?;
        if a.len() > 0 && b.len() > 0 {
            (self.dispatch)(&*self.mini_op, a, b, &output)
                .with_context(|| format!("Error while dispatching eval for {}", self.name()))?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for GpuBinOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.resolve_output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}

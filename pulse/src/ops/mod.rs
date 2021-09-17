use crate::internal::*;
use tract_pulse_opl::ops::Delay;

pub mod array;
pub mod cnn;
pub mod delay;
pub mod downsample;
pub mod dummy;
pub mod matmul;
pub mod qmatmul;
pub mod scan;
pub mod slice;
pub mod source;

pub(crate) fn sync_inputs(
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<TVec<OutletId>> {
    let delay = node
        .inputs
        .iter()
        .map(|input| target.outlet_fact(mapping[input]).unwrap().delay)
        .max()
        .unwrap();
    let mut inputs = tvec!();
    for input in &node.inputs {
        let mut input = mapping[input];
        let fact = target.outlet_fact(input)?.clone();
        if fact.delay < delay {
            let add_delay = delay - fact.delay;
            let delay_axis = fact.axis;
            input = target.wire_node(
                format!("{}.Delay", &*node.name),
                Delay::new_typed(&fact.into(), delay_axis, add_delay, 0),
                &[input],
            )?[0];
        }
        inputs.push(input);
    }
    Ok(inputs)
}


register_all_mod!(
    array,
    cnn,
    downsample,
    matmul,
    qmatmul,
    scan,
    source
);

pub struct OpPulsifier {
    pub type_id: std::any::TypeId,
    pub name: &'static str,
    pub func: fn(
        &TypedModel,
        &TypedNode,
        &mut PulsedModel,
        &HashMap<OutletId, OutletId>,
        usize,
    ) -> TractResult<Option<TVec<OutletId>>>,
}

impl OpPulsifier {
    pub fn inventory() -> HashMap<TypeId, OpPulsifier> {
        let mut inventory = HashMap::default();
        register_all(&mut inventory);
        inventory
    }
}

pub trait PulsedOp:
    Op
    + fmt::Debug
    + tract_core::dyn_clone::DynClone
    + Send
    + Sync
    + 'static
    + Downcast
    + EvalOp
    + DynHash
{
    /// Reinterpret the PulsedOp as an Op.
    fn as_op(&self) -> &dyn Op;

    /// Reinterpret the PulsedOp as an Op, mutably.
    fn as_op_mut(&mut self) -> &mut dyn Op;

    /// Reinterpret the PulsedOp as an TypedOp.
    fn to_typed(&self) -> Box<dyn TypedOp>;

    /// Deduce output facts from input facts.
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>>;
}

tract_core::dyn_clone::clone_trait_object!(PulsedOp);

impl Hash for Box<dyn PulsedOp> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.type_id(), state);
        self.dyn_hash(state)
    }
}

impl<O: PulsedOp> From<O> for Box<dyn PulsedOp> {
    fn from(it: O) -> Box<dyn PulsedOp> {
        Box::new(it)
    }
}

impl AsMut<dyn Op> for Box<dyn PulsedOp> {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl AsRef<dyn Op> for dyn PulsedOp {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsRef<dyn Op> for Box<dyn PulsedOp> {
    fn as_ref(&self) -> &dyn Op {
        self.as_op()
    }
}

impl AsMut<dyn Op> for dyn PulsedOp {
    fn as_mut(&mut self) -> &mut dyn Op {
        self.as_op_mut()
    }
}

impl std::fmt::Display for Box<dyn PulsedOp> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}", self.name())
    }
}

impl<'a> From<&'a Box<dyn PulsedOp>> for Box<dyn TypedOp> {
    fn from(op: &'a Box<dyn PulsedOp>) -> Box<dyn TypedOp> {
        op.to_typed()
    }
}

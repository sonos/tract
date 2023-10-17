use std::any::Any;
use std::sync::RwLock;

use crate::internal::*;
use lazy_static::lazy_static;
use tract_pulse_opl::ops::Delay;

pub mod array;
pub mod cnn;
pub mod delay;
pub mod downsample;
pub mod dummy;
pub mod mask;
pub mod scan;
pub mod slice;
pub mod source;

pub(crate) fn sync_inputs(
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<TVec<OutletId>> {
    let mut max_delay = 0;
    for input in &node.inputs {
        let fact = target.outlet_fact(mapping[input])?;
        if let Some(stream) = &fact.stream {
            max_delay = max_delay.max(stream.delay);
        }
    }
    let mut inputs = tvec!();
    for input in &node.inputs {
        let mut input = mapping[input];
        let fact = target.outlet_fact(input)?.clone();
        if let Some(stream) = &fact.stream {
            if stream.delay < max_delay {
                let add_delay = max_delay - stream.delay;
                let delay_axis = stream.axis;
                input = target.wire_node(
                    format!("{}.Delay", &*node.name),
                    Delay::new_typed(&fact.into(), delay_axis, add_delay, 0),
                    &[input],
                )?[0];
            }
        }
        inputs.push(input);
    }
    Ok(inputs)
}

register_all_mod!(array, cnn, downsample, scan, source);

type PulsifierFn = fn(
    &TypedModel,
    &TypedNode,
    &mut PulsedModel,
    &HashMap<OutletId, OutletId>,
    &Symbol,
    &TDim,
) -> TractResult<Option<TVec<OutletId>>>;

pub struct OpPulsifier {
    pub type_id: std::any::TypeId,
    pub name: &'static str,
    pub func: PulsifierFn,
}

impl OpPulsifier {
    pub fn inventory() -> Arc<RwLock<HashMap<TypeId, OpPulsifier>>> {
        lazy_static! {
            static ref INVENTORY: Arc<RwLock<HashMap<TypeId, OpPulsifier>>> = {
                let mut it = HashMap::default();
                register_all(&mut it);
                Arc::new(RwLock::new(it))
            };
        };
        (*INVENTORY).clone()
    }

    pub fn register<T: Any>(func: PulsifierFn) -> TractResult<()> {
        let inv = Self::inventory();
        let mut inv = inv.write().map_err(|e| anyhow!("Fail to lock inventory {e}"))?;
        inv.insert(
            std::any::TypeId::of::<T>(),
            OpPulsifier {
                type_id: std::any::TypeId::of::<T>(),
                name: std::any::type_name::<T>(),
                func,
            },
        );
        Ok(())
    }

    pub fn pulsify(
        source: &TypedModel,
        node: &TypedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        symbol: &Symbol,
        pulse: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        let inv = Self::inventory();
        let inv = inv.read().map_err(|e| anyhow!("Fail to lock inventory {e}"))?;
        if let Some(pulsifier) = inv.get(&(*node.op).type_id()) {
            if let Some(pulsified) = (pulsifier.func)(source, node, target, mapping, symbol, pulse)?
            {
                return Ok(Some(pulsified));
            }
        }
        Ok(None)
    }
}

pub trait PulsedOp:
    Op + fmt::Debug + tract_core::dyn_clone::DynClone + Send + Sync + 'static + Downcast + EvalOp
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

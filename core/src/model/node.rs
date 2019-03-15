use super::*;
use crate::ops::Op;

pub type TVec<T> = ::smallvec::SmallVec<[T; 4]>;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct Node<TI: TensorInfo> {
    pub id: usize,
    pub name: String,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: Box<Op>,
    pub outputs: TVec<OutletFact<TI>>,
}

impl<TI: TensorInfo> Node<TI> {
    pub fn op(&self) -> &Op {
        &*self.op
    }

    pub fn op_as<O: Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }

    pub fn op_is<O: Op>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    pub fn same_as(&self, other: &Node<TI>) -> bool {
        self.inputs == other.inputs && self.op.same_as(other.op.as_ref())
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletFact<TI: TensorInfo> {
    pub fact: TI,
    pub successors: TVec<InletId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletId {
    pub node: usize,
    pub slot: usize,
}

impl OutletId {
    pub fn new(node: usize, slot: usize) -> OutletId {
        OutletId { node, slot }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct InletId {
    pub node: usize,
    pub slot: usize,
}

impl InletId {
    pub fn new(node: usize, slot: usize) -> InletId {
        InletId { node, slot }
    }
}


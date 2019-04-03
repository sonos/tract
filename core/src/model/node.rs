use super::*;
use crate::ops::Op;
use itertools::Itertools;
use std::fmt;

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

impl<TI: TensorInfo> fmt::Display for Node<TI> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "#{} \"{}\" {}", self.id, self.name, self.op().name())
    }
}

impl<TI: TensorInfo> Node<TI> {
    pub fn op(&self) -> &Op {
        &*self.op
    }

    pub fn op_as<O: Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }

    pub fn op_as_mut<O: Op>(&mut self) -> Option<&mut O> {
        self.op.downcast_mut::<O>()
    }

    pub fn op_is<O: Op>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    pub fn same_as(&self, other: &Node<TI>) -> bool {
        self.inputs == other.inputs && self.op.same_as(other.op.as_ref())
    }
}

#[derive(Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletFact<TI: TensorInfo> {
    pub fact: TI,
    pub successors: TVec<InletId>,
}

impl<TI: TensorInfo> fmt::Debug for OutletFact<TI> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "{:?} {}",
            self.fact,
            self.successors.iter().map(|o| format!("{:?}", o)).join(" ")
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

impl fmt::Debug for OutletId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}/{}>", self.node, self.slot)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

impl fmt::Debug for InletId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, ">{}/{}", self.node, self.slot)
    }
}

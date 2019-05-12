use super::*;
use crate::ops::Op;
use itertools::Itertools;
use std::fmt;

/// A Smallvec instantiation with 4 embeddable values.
///
/// Used about everywhere in tract, for node inputs and outputs, or 
/// tensor dimensions.
pub type TVec<T> = ::smallvec::SmallVec<[T; 4]>;

/// A Node in an Model.
///
/// Parameterized by a TensorInfo implementation matching the one used in the
/// model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct Node<TI: TensorInfo> {
    /// node id in the model
    ///
    /// Caution: this id will not be persistent during networks transformation
    pub id: usize,
    /// name of the node
    ///
    /// This will usually come from the importing framework. `tract`
    /// transformation try to maintain the names accross transformations.
    pub name: String,
    /// A list of incoming tensors, identified by the node outlet that creates
    /// them.
    pub inputs: Vec<OutletId>,
    /// The actual operation the node performs.
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: Box<Op>,
    /// List of ouputs, with their descendant and tensor type information.
    pub outputs: TVec<OutletFact<TI>>,
}

impl<TI: TensorInfo> fmt::Display for Node<TI> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "#{} \"{}\" {}", self.id, self.name, self.op().name())
    }
}

impl<TI: TensorInfo> Node<TI> {
    /// Access the op of the node
    pub fn op(&self) -> &Op {
        &*self.op
    }

    /// Try to downcast the node operation to O.
    pub fn op_as<O: Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }

    /// Try to downcast the node operation to O.
    pub fn op_as_mut<O: Op>(&mut self) -> Option<&mut O> {
        self.op.downcast_mut::<O>()
    }

    /// Check if the node operation is of type O.
    pub fn op_is<O: Op>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    /// Check that this node produce the same outputs as `other`.
    pub fn same_as(&self, other: &Node<TI>) -> bool {
        self.inputs == other.inputs && self.op.same_as(other.op.as_ref())
    }
}

/// Information for each outlet of a node
#[derive(Clone, Default)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletFact<TI: TensorInfo> {
    /// the tensor type information
    pub fact: TI,
    /// where this outlet is used.
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

/// Identifier for a node output in the graph.
///
/// This happens to be a unique identifier of any variable tensor in the graph
/// (as the graph typically connect one single node output to one or several
/// inputs slots)
#[derive(Clone, Copy, PartialEq, Eq, Hash, new)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct OutletId {
    /// node identifier in the graph
    pub node: usize,
    /// rank of the input in the node
    pub slot: usize,
}

impl fmt::Debug for OutletId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}/{}>", self.node, self.slot)
    }
}

/// Identifier for a node input in the graph.
#[derive(Clone, Copy, PartialEq, Eq, Hash, new)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct InletId {
    /// node identifier in the graph
    pub node: usize,
    /// rank of the input in the node
    pub slot: usize,
}

impl fmt::Debug for InletId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, ">{}/{}", self.node, self.slot)
    }
}

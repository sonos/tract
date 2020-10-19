use super::*;
use crate::ops::Op;
use itertools::Itertools;
use std::fmt;
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// A Node in an Model.
///
/// Parameterized by a Fact implementation matching the one used in the
/// model.
#[derive(Debug, Clone, Educe)]
#[educe(Hash)]
pub struct Node<F: Fact + Hash, O: Hash> {
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
    pub op: O,
    /// List of ouputs, with their descendant and tensor type information.
    pub outputs: TVec<Outlet<F>>,
}

impl<F: Fact + Hash, O: Hash + std::fmt::Display> fmt::Display for Node<F, O> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "#{} \"{}\" {}", self.id, self.name, self.op)
    }
}

impl<F, NodeOp> Node<F, NodeOp>
where
    F: Fact + Hash,
    NodeOp: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + AsMut<dyn Op> + Hash,
{
    /// Access the op of the node
    pub fn op(&self) -> &dyn Op {
        self.op.as_ref()
    }

    /// Try to downcast the node operation to O.
    pub fn op_as<O: Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }

    /// Try to downcast the node operation to O.
    pub fn op_as_mut<O: Op>(&mut self) -> Option<&mut O> {
        self.op.as_mut().downcast_mut::<O>()
    }

    /// Check if the node operation is of type O.
    pub fn op_is<O: Op>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    /// Check that this node produce the same outputs as `other`.
    pub fn same_as(&self, other: &Node<F, NodeOp>) -> bool {
        self.inputs == other.inputs && self.op().same_as(other.op())
    }
}

/// Information for each outlet of a node
#[derive(Clone, Default, Educe)]
#[educe(Hash)]
pub struct Outlet<F: Fact + Hash> {
    /// the tensor type information
    pub fact: F,
    /// where this outlet is used.
    pub successors: TVec<InletId>,
}

impl<F: Fact + Hash> fmt::Debug for Outlet<F> {
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, new)]
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

impl From<usize> for OutletId {
    fn from(node: usize) -> OutletId {
        OutletId::new(node, 0)
    }
}

impl From<(usize, usize)> for OutletId {
    fn from(pair: (usize, usize)) -> OutletId {
        OutletId::new(pair.0, pair.1)
    }
}

/// Identifier for a node input in the graph.
#[derive(Clone, Copy, PartialEq, Eq, Hash, new, Ord, PartialOrd)]
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

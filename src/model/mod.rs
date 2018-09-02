use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::str;

pub mod tf;
mod order;
pub use self::order::eval_order_for_nodes;

use {ops, Result};

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub name: String,
    pub op_name: String,
    pub inputs: Vec<OutletId>,
    pub op: Box<ops::Op>,
}

impl Node {
    pub fn op(&self) -> &ops::Op {
        &*self.op
    }
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
pub struct InletId {
    pub node: usize,
    pub inlet: usize,
}

impl InletId {
    pub fn new(node: usize, inlet: usize) -> InletId {
        InletId { node, inlet }
    }
}

pub type TVec<T> = ::smallvec::SmallVec<[T; 4]>;

/*
pub struct OutletIndex<T>(Vec<TVec<T>>);

impl<T:Default> OutletIndex<T> {
    pub fn at(&mut self, idx: OutletId) -> &mut T {
        while self.0.len() < idx.node {
            self.0.push(tvec!());
        }
        let tv = self.0[idx.node];
        while tv.len() < idx.slot {
            tv.push(T::default())
        }
        &mut tv[idx.slot]
    }
}

impl<T> OutletIndex<T> {
    pub fn get(&self, idx: OutletId) -> Option<&T> {
        self.get(idx.node).and_then(|v| v.get(idx.slot))
    }
}

impl<T> ::std::ops::Index<OutletId> for OutletIndex<T> {
    type Output = T;
    fn index(&self, idx: OutletId) -> &T {
        self.get(idx.node).and_then(idx.port)
    }
}
*/

/// Model is Tfdeploy workhouse. It wraps a protobuf tensorflow model,
/// and runs the inference interpreter.
#[derive(Clone, Debug)]
pub struct RawModel {
    pub nodes: Vec<Node>,
    pub nodes_by_name: HashMap<String, usize>,
}

impl RawModel {
    pub fn node_by_name(&self, name: &str) -> Result<&Node> {
        let id: &usize = self.nodes_by_name
            .get(name)
            .ok_or_else(|| format!("Node named {} not found", name))?;
        Ok(&self.nodes[*id])
    }

    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| &*s.name).collect()
    }

    pub fn nodes(&self) -> &[Node] {
        &*self.nodes
    }
}

#[derive(Debug, Clone)]
pub struct Model(pub Arc<RawModel>);

impl Model {
    pub fn analyser(&self, output: &str) -> Result<::analyser::Analyser> {
        ::analyser::Analyser::new(&self, output)
    }
}

impl Deref for Model {
    type Target = RawModel;
    fn deref(&self) -> &RawModel {
        &*self.0
    }
}


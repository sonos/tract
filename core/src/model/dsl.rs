use crate::internal::*;
use crate::ops::dummy::Dummy;
use crate::pulse::PulsedFact;
use std::fmt;

pub use super::{InletId, ModelImpl, Node, OutletId};

pub trait ModelSpecialOps<F, O>
where
    F: Fact + Clone + 'static + Hash,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    /// Adds a source op to the network.
    ///
    /// The model will assume this is an input.
    fn add_source(&mut self, name: impl Into<String>, fact: F) -> TractResult<OutletId>;
    fn is_source(op: &dyn Op) -> bool;

    fn create_dummy(&self) -> O;
}

impl ModelSpecialOps<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn add_source(&mut self, name: impl Into<String>, fact: TypedFact) -> TractResult<OutletId> {
        let id =
            self.add_node(name, crate::ops::source::TypedSource::new(fact.clone()), tvec!(fact))?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }

    fn is_source(op: &dyn Op) -> bool {
        op.downcast_ref::<crate::ops::source::TypedSource>().is_some()
    }

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Dummy::new())
    }
}

impl ModelSpecialOps<PulsedFact, Box<dyn TypedOp>> for PulsedModel {
    fn add_source(&mut self, name: impl Into<String>, fact: PulsedFact) -> TractResult<OutletId> {
        let id =
            self.add_node(name, crate::ops::source::PulsedSource::new(fact.clone()), tvec!(fact))?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }

    fn is_source(op: &dyn Op) -> bool {
        op.downcast_ref::<crate::ops::source::PulsedSource>().is_some()
    }

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Dummy::new())
    }
}

/// Extensions on ModelImpl to explore and build graph models more easily.
pub trait ModelDsl<F, O>
where
    F: Fact + Clone + 'static + Hash,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    /// Find the lone precursor of a node, if applicable.
    fn single_prec(&self, id: usize) -> TractResult<Option<&BaseNode<F, O>>>;
    /// Find the count-th precursor of a node `id` in a chain of single tensor
    /// operation, if applicable.
    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<F, O>>>;
    /// Find the lone succesor of a node, if applicable.
    fn single_succ(&self, id: usize) -> TractResult<Option<&BaseNode<F, O>>>;
    /// Find the count-th successor of a node `id` in a chain of single tensor
    /// operation, if applicable.
    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<F, O>>>;
}

impl<F, O> ModelDsl<F, O> for ModelImpl<F, O>
where
    F: Fact + Clone + 'static + Hash,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    fn single_prec(&self, id: usize) -> TractResult<Option<&BaseNode<F, O>>> {
        let node = &self.nodes()[id];
        if node.inputs.len() != 1 {
            return Ok(None);
        }
        let prec = &self.nodes()[node.inputs[0].node];
        if prec.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return Ok(None);
        }
        Ok(Some(prec))
    }

    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<F, O>>> {
        let mut node = self.node(id);
        for _ in 0..count {
            if let Some(next) = self.single_prec(node.id)? {
                node = next
            } else {
                return Ok(None);
            }
        }
        Ok(Some(node))
    }

    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<F, O>>> {
        let mut node = self.node(id);
        for _ in 0..count {
            if let Some(next) = self.single_succ(node.id)? {
                node = next
            } else {
                return Ok(None);
            }
        }
        Ok(Some(node))
    }

    fn single_succ(&self, id: usize) -> TractResult<Option<&BaseNode<F, O>>> {
        let node = &self.nodes()[id];
        if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
            return Ok(None);
        }
        let succ = node.outputs[0].successors[0];
        let succ = &self.nodes()[succ.node];
        if succ.inputs.len() != 1 {
            return Ok(None);
        }
        Ok(Some(succ))
    }
}

/// Extension to add constants to model that tolerates them.
pub trait ModelDslConst {
    /// Add a constant node to the graph.
    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId>;
}

impl<F: Fact + Clone + 'static, O> ModelDslConst for ModelImpl<F, O>
where
    F: Fact + Clone + 'static + From<Arc<Tensor>> + Hash,
    O: fmt::Debug
        + fmt::Display
        + From<crate::ops::konst::Const>
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + Hash
        + 'static,
{
    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        let fact = F::from(v.clone());
        self.add_node(name, crate::ops::konst::Const::new(v), tvec!(fact)).map(|id| id.into())
    }
}

pub trait ModelWireNode<F, O>
where
    F: Fact + Clone + 'static + Hash,
    O: fmt::Debug + fmt::Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>>;
}

impl ModelWireNode<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let name = name.into();
        let output_facts = {
            let input_facts =
                inputs.iter().map(|o| self.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            if input_facts.iter().all(|f| f.konst.is_some()) && op.as_stateless().is_some() {
                let tensors =
                    input_facts.iter().map(|f| f.konst.clone().unwrap()).collect::<TVec<_>>();
                let outputs = op.as_stateless().unwrap().eval(tensors)?;
                outputs.into_iter().map(|t| TypedFact::from(t)).collect()
            } else {
                op.output_facts(&*input_facts).chain_err(|| format!("wiring {}", name))?
            }
        };
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }
}

impl ModelWireNode<PulsedFact, Box<dyn PulsedOp>> for PulsedModel {
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn PulsedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts = {
            let input_facts =
                inputs.iter().map(|o| self.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            op.pulsed_output_facts(&*input_facts)?
        };
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }
}

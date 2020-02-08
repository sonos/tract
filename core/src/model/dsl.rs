use crate::internal::*;
use crate::infer::*;
use crate::ops::dummy::Dummy;
use crate::pulse::PulsedFact;
use std::convert::TryFrom;
use std::fmt::{Debug, Display};

pub use super::{InletId, ModelImpl, Node, OutletId};

pub trait ModelSpecialOps<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    /// Adds a source op to the network.
    ///
    /// The model will assume this is an input.
    fn add_source(&mut self, name: impl Into<String>, fact: TI) -> TractResult<OutletId>;

    fn create_dummy(&self) -> O;
}

impl ModelSpecialOps<InferenceFact, Box<dyn InferenceOp>> for InferenceModel {
    fn add_source(
        &mut self,
        name: impl Into<String>,
        fact: InferenceFact,
    ) -> TractResult<OutletId> {
        let id = self.add_node(name, crate::hir::source::Source::new(), tvec!(fact))?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }

    fn create_dummy(&self) -> Box<dyn InferenceOp> {
        Box::new(Dummy::new())
    }
}

impl ModelSpecialOps<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn add_source(&mut self, name: impl Into<String>, fact: TypedFact) -> TractResult<OutletId> {
        let id =
            self.add_node(name, crate::ops::source::TypedSource::new(fact.clone()), tvec!(fact))?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
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

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Dummy::new())
    }
}

/// Extensions on ModelImpl to explore and build graph models more easily.
pub trait ModelDsl<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    /// Find the lone precursor of a node, if applicable.
    fn single_prec(&self, id: usize) -> TractResult<Option<&BaseNode<TI, O>>>;
    /// Find the count-th precursor of a node `id` in a chain of single tensor
    /// operation, if applicable.
    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<TI, O>>>;
    /// Find the lone succesor of a node, if applicable.
    fn single_succ(&self, id: usize) -> TractResult<Option<&BaseNode<TI, O>>>;
    /// Find the count-th successor of a node `id` in a chain of single tensor
    /// operation, if applicable.
    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<TI, O>>>;
}

impl<TI, O> ModelDsl<TI, O> for ModelImpl<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn single_prec(&self, id: usize) -> TractResult<Option<&BaseNode<TI, O>>> {
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

    fn single_prec_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<TI, O>>> {
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

    fn single_succ_at(&self, id: usize, count: usize) -> TractResult<Option<&BaseNode<TI, O>>> {
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

    fn single_succ(&self, id: usize) -> TractResult<Option<&BaseNode<TI, O>>> {
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

impl<TI: Fact + Clone + 'static, O, E> ModelDslConst for ModelImpl<TI, O>
where
    TractError: From<E>,
    TI: Fact + Clone + 'static + for<'a> TryFrom<&'a InferenceFact, Error = E>,
    O: Debug
        + Display
        + From<crate::ops::konst::Const>
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + 'static,
{
    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        let fact = TI::try_from(&InferenceFact::from(&*v))?;
        self.add_node(name, crate::ops::konst::Const::new(v), tvec!(fact)).map(|id| id.into())
    }
}

pub trait ModelWireNode<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>>;
}

impl ModelWireNode<InferenceFact, Box<dyn InferenceOp>> for InferenceModel {
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts: TVec<InferenceFact> =
            (0..op.nboutputs()?).map(|_| InferenceFact::default()).collect();
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }
}

impl ModelWireNode<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn TypedOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts = {
            let input_facts =
                inputs.iter().map(|o| self.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
            if input_facts.iter().all(|f| f.konst.is_some()) && op.as_stateless().is_some() {
                let tensors =
                    input_facts.iter().map(|f| f.konst.clone().unwrap()).collect::<TVec<_>>();
                let outputs = op.as_stateless().unwrap().eval(tensors)?;
                outputs.into_iter().map(|t| TypedFact::from(t)).collect()
            } else {
                op.output_facts(&*input_facts)?
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

use crate::internal::*;
use crate::ops::dummy::Dummy;
use crate::pulse::PulsedTensorFact;
use std::convert::TryFrom;
use std::fmt::{Debug, Display};

pub use super::{InletId, ModelImpl, Node, OutletId};

pub trait ModelSpecialOps<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    /// Adds a source op to the network.
    ///
    /// The model will assume this is an input.
    fn add_source(&mut self, name: impl Into<String>, fact: TI) -> TractResult<usize>;

    fn create_dummy(&self) -> O;
}

impl ModelSpecialOps<TensorFact, Box<dyn InferenceOp>> for InferenceModel {
    fn add_source(&mut self, name: impl Into<String>, fact: TensorFact) -> TractResult<usize> {
        let id = self.add_node(name, crate::ops::source::Source::new(), tvec!(fact))?;
        self.inputs.push(OutletId::new(id, 0));
        Ok(id)
    }

    fn create_dummy(&self) -> Box<dyn InferenceOp> {
        Box::new(Dummy::new())
    }
}

impl ModelSpecialOps<TypedTensorInfo, Box<dyn TypedOp>> for TypedModel {
    fn add_source(&mut self, name: impl Into<String>, fact: TypedTensorInfo) -> TractResult<usize> {
        let id =
            self.add_node(name, crate::ops::source::TypedSource::new(fact.clone()), tvec!(fact))?;
        self.inputs.push(OutletId::new(id, 0));
        Ok(id)
    }

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Dummy::new())
    }
}

impl ModelSpecialOps<PulsedTensorFact, Box<dyn TypedOp>> for PulsedModel {
    fn add_source(
        &mut self,
        name: impl Into<String>,
        fact: PulsedTensorFact,
    ) -> TractResult<usize> {
        let id = self.add_node(
            name,
            crate::ops::source::TypedSource::new(TypedTensorInfo::dt_shape(fact.dt, &*fact.shape)?),
            tvec!(fact),
        )?;
        self.inputs.push(OutletId::new(id, 0));
        Ok(id)
    }

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(Dummy::new())
    }
}

/// Extensions on ModelImpl to explore and build graph models more easily.
pub trait ModelDsl<TI, O>
where
    TI: TensorInfo + Clone + 'static,
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

    /// Chain a node to the latest inserted node.
    ///
    /// * creates a node with name and op
    /// * connect the 0-th input of the new node to the 0-th outlet of the
    /// latest previously inserted node.
    fn chain(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        facts: TVec<TI>,
    ) -> TractResult<usize>;

    /// Chain a node to an arbitrary node.
    ///
    /// * creates a node with name and op
    /// * connect the 0-th input of the new node to `tap`
    fn chain_after(
        &mut self,
        tap: OutletId,
        name: impl Into<String>,
        op: impl Into<O>,
        facts: TVec<TI>,
    ) -> TractResult<usize>;
}

impl<TI, O> ModelDsl<TI, O> for ModelImpl<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn chain(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        facts: TVec<TI>,
    ) -> TractResult<usize> {
        let previous_id = self.nodes().len() - 1;
        self.chain_after(OutletId::new(previous_id, 0), name, op.into(), facts)
    }

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

    fn chain_after(
        &mut self,
        tap: OutletId,
        name: impl Into<String>,
        op: impl Into<O>,
        facts: TVec<TI>,
    ) -> TractResult<usize> {
        let id = self.add_node(name, op, facts)?;
        self.add_edge(tap, InletId::new(id, 0))?;
        Ok(id)
    }
}

/// Extension to add constants to model that tolerates them.
pub trait ModelDslConst {
    /// Add a constant node to the graph.
    fn add_const(&mut self, name: impl Into<String>, v: impl IntoArcTensor) -> TractResult<OutletId>;
    /// Add a constant node to the graph and connect its output to `inlet`.
    fn plug_const(
        &mut self,
        inlet: InletId,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<()>;
}

impl<TI: TensorInfo + Clone + 'static, O, E> ModelDslConst for ModelImpl<TI, O>
where
    TractError: From<E>,
    TI: TensorInfo + Clone + 'static + TryFrom<TensorFact, Error = E>,
    O: Debug
        + Display
        + From<crate::ops::konst::Const>
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + 'static,
{
    fn add_const(&mut self, name: impl Into<String>, v: impl IntoArcTensor) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        let fact = TI::try_from(TensorFact::from(v.clone()))?;
        self.add_node(name, crate::ops::konst::Const::new(v), tvec!(fact)).map(|id| id.into())
    }
    fn plug_const(
        &mut self,
        inlet: InletId,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<()> {
        let cst = self.add_const(name, v)?;
        self.add_edge(cst, inlet)?;
        Ok(())
    }
}

/*
impl ModelDslConst for super::TypedModel {
    fn add_const(&mut self, name: impl Into<String>, v: impl IntoArcTensor) -> TractResult<usize> {
        let v = v.into_arc_tensor();
        let facts = tvec!(v.clone().into());
        self.add_node(name, crate::ops::konst::Const::new(v), facts)
    }
    fn plug_const(
        &mut self,
        inlet: InletId,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<()> {
        let cst = self.add_const(name, v)?;
        self.add_edge(OutletId::new(cst, 0), inlet)?;
        Ok(())
    }
}
*/

pub trait ModelDslInfer: ModelDsl<TensorFact, Box<dyn InferenceOp>> {
    /// Add a source with no tensor information.
    fn add_source_default(&mut self, name: impl Into<String>) -> TractResult<usize>;

    /// Add a node without tensor information.
    fn add_node_default(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
    ) -> TractResult<usize>;

    /// Chain a node without tensor information.
    fn chain_default(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
    ) -> TractResult<usize>;
}

impl ModelDslInfer for super::InferenceModel {
    fn add_source_default(&mut self, name: impl Into<String>) -> TractResult<usize> {
        self.add_source(name, TensorFact::default())
    }

    fn add_node_default(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
    ) -> TractResult<usize> {
        self.add_node(name, op, tvec!(TensorFact::default()))
    }

    fn chain_default(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
    ) -> TractResult<usize> {
        self.chain(name, op, tvec!(TensorFact::default()))
    }
}

pub trait ModelWireNode<TI, O>
where
    TI: TensorInfo + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>>;
}

impl ModelWireNode<TensorFact, Box<dyn InferenceOp>> for InferenceModel {
    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<Box<dyn InferenceOp>>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let op = op.into();
        let output_facts: TVec<TensorFact> =
            (0..op.nboutputs()?).map(|_| TensorFact::default()).collect();
        let id = self.add_node(name, op, output_facts)?;
        inputs
            .iter()
            .enumerate()
            .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
        Ok(self.node(id).outputs.iter().enumerate().map(|(ix, _)| OutletId::new(id, ix)).collect())
    }
}

impl ModelWireNode<TypedTensorInfo, Box<dyn TypedOp>> for TypedModel {
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
                outputs.into_iter().map(|t| TypedTensorInfo::from(t)).collect()
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

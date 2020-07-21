use crate::internal::*;
use crate::ops::dummy::Dummy;
use crate::pulse::PulsedFact;
use std::fmt;

pub use super::{InletId, ModelImpl, Node, OutletId};

pub trait ModelDSL<F, O>
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

    fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>>;
}

impl ModelDSL<TypedFact, Box<dyn TypedOp>> for TypedModel {
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

impl ModelDSL<PulsedFact, Box<dyn PulsedOp>> for PulsedModel {
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

    fn create_dummy(&self) -> Box<dyn PulsedOp> {
        Box::new(Dummy::new())
    }

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

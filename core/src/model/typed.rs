use crate::errors::TractResultExt;
use crate::model::*;
use crate::ops;
use crate::ops::invariants;
use crate::plan::{SimplePlan, SimpleState};
use crate::TractResult;

/// A model with completely determined types and shapes.
pub type TypedModel = Graph<TypedFact, Box<dyn TypedOp>>;
/// Node for TypedModel graph
pub type TypedNode = BaseNode<TypedFact, Box<dyn TypedOp>>;
/// A ModelPatch for TypedModel.
pub type TypedModelPatch = ModelPatch<TypedFact, Box<dyn TypedOp>>;
/// An execution plan for TypedModel.
pub type TypedSimplePlan<M> = SimplePlan<TypedFact, Box<dyn TypedOp>, M>;
/// A runnable TypedModel (new name for SimplePlan).
pub type TypedRunnableModel<M> = RunnableModel<TypedFact, Box<dyn TypedOp>, M>;
/// An execution state for TypedModel.
pub type TypedSimpleState<M, P> = SimpleState<TypedFact, Box<dyn TypedOp>, M, P>;

/// A runnable model with fixed inputs and outputs.
pub type RunnableModel<F, O, M> = SimplePlan<F, O, M>;

impl SpecialOps<TypedFact, Box<dyn TypedOp>> for TypedModel {
    fn is_source(op: &Box<dyn TypedOp>) -> bool {
        op.as_op().downcast_ref::<ops::source::TypedSource>().is_some()
    }

    fn create_dummy(&self) -> Box<dyn TypedOp> {
        Box::new(crate::ops::dummy::Dummy::new())
    }

    fn create_source(&self, fact: TypedFact) -> Box<dyn TypedOp> {
        Box::new(crate::ops::source::TypedSource::new(fact))
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

impl TypedModel {
    pub fn signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn into_optimized(self) -> TractResult<TypedModel> {
        self.declutter()?.optimize()
    }

    /// Perform declutter passes on the network.
    pub fn declutter(&self) -> TractResult<TypedModel> {
        let mut model = self.clone();
        let mut seen = std::collections::HashSet::new();
        for i in 0.. {
            model = model.compact()?;
            let mut done_something_this_time = false;
            for p in crate::optim::declutter() {
                if p.pass(&mut model)? {
                    done_something_this_time = true;
                    debug!("done_something at pass {} in {:?}", i, p);
                }
                done_something_this_time = done_something_this_time || p.pass(&mut model)?;
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                }
            }
            if !done_something_this_time {
                return Ok(model);
            }
            if i < 10 {
                continue;
            }
            let sig = model.signature();
            if seen.contains(&sig) {
                return Ok(model);
            }
            seen.insert(sig);
            model = model.compact()?;
        }
        unreachable!()
    }

    pub fn concretize_stream_dim(&self, dim: usize) -> TractResult<TypedModel> {
        use crate::model::translator::Translate;
        #[derive(Debug)]
        struct ConcretizeStreamDim(usize);
        impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for ConcretizeStreamDim {
            fn translate_node(
                &self,
                source: &TypedModel,
                node: &TypedNode,
                target: &mut TypedModel,
                mapping: &HashMap<OutletId, OutletId>,
            ) -> TractResult<TVec<OutletId>> {
                node.op.concretize_stream_dim(source, node, target, mapping, self.0)
            }
        }
        ConcretizeStreamDim(dim).translate_model(&self)
    }

    /// Translate the graph to locally optimized operators (LIR or MIR ops).
    pub fn optimize(self) -> TractResult<TypedModel> {
        let mut model = self;
        loop {
            let mut done_something = false;
            for p in crate::optim::codegen() {
                done_something = done_something || p.pass(&mut model)?;
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                }
            }
            if !done_something {
                break;
            }
            model = model.compact()?;
        }
        Ok(model)
    }

    pub fn invariants(&self) -> TractResult<invariants::Invariants> {
        invariants::for_model(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        fn is_sync<T: Sync>() {}
        is_sync::<TypedModel>();
    }
}

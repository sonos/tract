use std::collections::HashMap;

use tract_core::ops::konst::Const;

use super::factoid::Factoid;
use super::{InferenceFact, InferenceModel, InferenceNode, InferenceOp};
use crate::internal::*;
use crate::prelude::TVec;

pub trait InferenceModelExt {
    /// Analyse all nodes of the graph.
    ///
    /// Will stop on first error unless `obstinate` is `true`.
    fn analyse(&mut self, obstinate: bool) -> TractResult<bool>;

    /// Perform early transformation before going typed.
    fn incorporate(self) -> TractResult<InferenceModel>;

    /// List OutletId with incomplete type information.
    ///
    /// Will stop on first error unless `obstinate` is `true`.
    fn missing_type_shape(&self) -> TractResult<Vec<OutletId>>;

    /// Eliminate seemingly dead branches of the graph.
    ///
    /// This may break stateful networks.
    fn eliminate_dead_branches(self) -> TractResult<InferenceModel>;

    /// Attempt full analyse and conversion to TypedModel.
    fn into_typed(self) -> TractResult<TypedModel>;

    /// Attempt full analyse, decluttering and mapping to optimized operations.
    ///
    /// This will work even if the network can not be normalized.
    fn into_optimized(self) -> TractResult<TypedModel>;
}

impl InferenceModelExt for InferenceModel {
    /// Analyse all nodes of the graph.
    ///
    /// Will stop on first error unless `obstinate` is `true`.
    fn analyse(&mut self, obstinate: bool) -> TractResult<bool> {
        super::analyser::Analyser::new(self).analyse_obstinate(obstinate)
    }

    /// Perform early transformation before going typed.
    fn incorporate(self) -> TractResult<InferenceModel> {
        let mut model = self;
        loop {
            let mut done_something = false;
            for p in crate::infer::optim::incorporate() {
                done_something = done_something || p.pass(&mut model)?;
                if cfg!(debug_assertions) {
                    model.check_edges()?;
                }
            }
            if !done_something {
                break;
            }
        }
        model = model.into_compact()?;
        model.analyse(false)?;
        Ok(model)
    }

    /// List OutletId with incomplete type information.
    ///
    /// Will stop on first error unless `obstinate` is `true`.
    fn missing_type_shape(&self) -> TractResult<Vec<OutletId>> {
        Ok(self
            .eval_order()?
            .iter()
            .flat_map(|&node| {
                self.nodes()[node]
                    .outputs
                    .iter()
                    .enumerate()
                    .map(move |(ix, outlet)| (OutletId::new(node, ix), outlet))
            })
            .filter(|(_, o)| !o.fact.datum_type.is_concrete() || !o.fact.shape.is_concrete())
            .map(|(id, _)| id)
            .collect())
    }

    /// Eliminate seemingly dead branches of the graph.
    ///
    /// This may break stateful networks.
    fn eliminate_dead_branches(self) -> TractResult<InferenceModel> {
        self.into_compact()
    }

    /// Attempt full analyse and conversion to TypedModel.
    fn into_typed(mut self) -> TractResult<TypedModel> {
        use tract_core::internal::translator::Translate;

        self.analyse(false)?;
        let m = self.incorporate()?;

        #[derive(Debug)]
        struct ToTypedTranslator;
        impl Translate<InferenceFact, Box<dyn InferenceOp>, TypedFact, Box<dyn TypedOp>>
            for ToTypedTranslator
        {
            fn translate_node(
                &self,
                source: &InferenceModel,
                node: &InferenceNode,
                target: &mut TypedModel,
                mapping: &HashMap<OutletId, OutletId>,
            ) -> TractResult<TVec<OutletId>> {
                if node.op.is_stateless()
                    && source.node_output_facts(node.id)?.iter().all(|f| f.value.is_concrete())
                {
                    (0..node.outputs.len())
                        .map(|ix| {
                            target.add_const(
                                format!("{}.{}", node.name, ix),
                                node.outputs[ix].fact.value.concretize().unwrap(),
                            )
                        })
                        .collect()
                } else {
                    let outputs = node.op.to_typed(source, node, target, mapping)?;
                    for output in &outputs {
                        let fact = target.outlet_fact(*output)?;
                        fact.consistent().with_context(|| {
                            format!(
                                "Checking oulet fact consistency for {:?}: {:?} after translating {:?}",
                                output,
                                fact, node.op,
                            )
                        })?;
                    }
                    Ok(outputs)
                }
            }
        }

        ToTypedTranslator.translate_model(&m)
    }

    /// Attempt full analyse, decluttering and mapping to optimized operations.
    ///
    /// This is meant for "simple" networks, where no special model
    /// transformation needs to happen. Aternaltively, use to_typed() and
    /// manipulate the TypedModel for more control.
    fn into_optimized(self) -> TractResult<TypedModel> {
        self.into_typed()?.into_optimized()
    }
}

impl SpecialOps<InferenceFact, Box<dyn InferenceOp>> for InferenceModel {
    fn is_source(op: &Box<dyn InferenceOp>) -> bool {
        op.as_op().downcast_ref::<crate::ops::source::Source>().is_some()
    }

    fn create_dummy(&self) -> Box<dyn InferenceOp> {
        Box::new(tract_core::ops::dummy::Dummy::new())
    }

    fn create_source(&self, _fact: InferenceFact) -> Box<dyn InferenceOp> {
        Box::new(crate::ops::source::Source::new())
    }

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

    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        for node in &self.nodes {
            if let Some(op) = node.op_as::<Const>() {
                if op.val() == &v {
                    return Ok(node.id.into());
                }
            }
        }
        let name = name.into();
        let fact = TypedFact::from(v.clone());
        self.add_node(name, crate::ops::konst::Const::new(v)?, tvec!(fact.into()))
            .map(|id| id.into())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        fn is_sync<T: Sync>() {}
        is_sync::<InferenceModel>();
    }
}

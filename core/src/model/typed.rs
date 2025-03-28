use crate::internal::*;
use crate::model::*;
use crate::ops;
use crate::ops::konst::Const;
use crate::optim::OptimizerSession;
use crate::plan::{FrozenSimpleState, SimplePlan, SimpleState};
use crate::transform::ModelTransform;
use tract_data::TooEarly;
use tract_linalg::block_quant::BlockQuantValue;
use tract_num_traits::Zero;

/// A model with completely determined types and shapes.
pub type TypedModel = Graph<TypedFact, Box<dyn TypedOp>>;
/// Node for TypedModel graph
pub type TypedNode = Node<TypedFact, Box<dyn TypedOp>>;
/// A ModelPatch for TypedModel.
pub type TypedModelPatch = ModelPatch<TypedFact, Box<dyn TypedOp>>;
/// An execution plan for TypedModel.
pub type TypedSimplePlan<M> = SimplePlan<TypedFact, Box<dyn TypedOp>, M>;
/// A runnable TypedModel (new name for SimplePlan).
pub type TypedRunnableModel<M> = RunnableModel<TypedFact, Box<dyn TypedOp>, M>;
/// An execution state for TypedModel.
pub type TypedSimpleState<M, P> = SimpleState<TypedFact, Box<dyn TypedOp>, M, P>;
/// An execution state for TypedModel, frozen (and Send).
pub type TypedFrozenSimpleState<M, P> = FrozenSimpleState<TypedFact, Box<dyn TypedOp>, M, P>;

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
        if let Some(konst) = op.downcast_ref::<Const>() {
            for node in &self.nodes {
                if node.op_as::<Const>().is_some_and(|other| other == konst) {
                    return Ok(tvec!(node.id.into()));
                }
            }
        }
        if self.nodes.iter().any(|n| n.name == name) {
            bail!("Duplicate node name: {name}");
        }
        {
            let input_facts = inputs
                .iter()
                .map(|o| self.outlet_fact(*o).cloned())
                .collect::<TractResult<TVec<_>>>()?;

            let input_facts: TVec<_> = input_facts.iter().collect();
            let mut output_facts = op
                .output_facts(&input_facts)
                .with_context(|| format!("in output_facts invocation for {name}: {}", op.name()))?;

            #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
            for o in &output_facts {
                o.consistent()?;
            }

            if op.is_stateless() && input_facts.len() > 0 {
                if let Some(tensors) = input_facts
                    .iter()
                    .map(|f| {
                        f.konst
                            .as_ref()
                            .filter(|k| k.volume() < 16 && !k.datum_type().is_opaque())
                            .cloned()
                            .map(|t| t.into_tvalue())
                    })
                    .collect::<Option<TVec<_>>>()
                {
                    if let Ok(outputs) = op.eval_with_session(&SessionState::default(), tensors) {
                        return outputs
                            .into_iter()
                            .enumerate()
                            .map(|(ix, o)| {
                                let name =
                                    if ix == 0 { name.clone() } else { format!("{name}.{ix}") };
                                self.wire_node(
                                    name.clone(),
                                    Const::new_with_opt_opaque_fact(
                                        o.into_tensor().into(),
                                        output_facts[ix].opaque_fact.clone(),
                                    )?,
                                    &[],
                                )
                                .with_context(|| format!("Eager const-folding {name}"))
                                .map(|vec| vec[0])
                            })
                            .collect::<TractResult<TVec<OutletId>>>();
                    }
                }
            }

            for fact in &mut output_facts {
                if fact.konst.is_none() && fact.shape.is_concrete() && fact.shape.volume().is_zero()
                {
                    let tensor =
                        Tensor::zero_dt(fact.datum_type, fact.shape.as_concrete().unwrap())?;
                    fact.konst = Some(tensor.into_arc_tensor());
                }
            }
            let id = self.add_node(&name, &op, output_facts)?;
            inputs
                .iter()
                .enumerate()
                .try_for_each(|(ix, i)| self.add_edge(*i, InletId::new(id, ix)))?;
            TractResult::Ok(
                self.node(id)
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(ix, _)| OutletId::new(id, ix))
                    .collect(),
            )
        }
        .with_context(|| format!("Wiring node \"{name}\", {op:?}"))
    }

    fn add_const(
        &mut self,
        name: impl Into<String>,
        v: impl IntoArcTensor,
    ) -> TractResult<OutletId> {
        let v = v.into_arc_tensor();
        for node in &self.nodes {
            if node.op_is::<Const>() && node.outputs[0].fact.konst.as_ref() == Some(&v) {
                return Ok(node.id.into());
            }
        }
        let mut fact = TypedFact::from(v.clone());
        let name = name.into();
        // this feel incredibly hackish and dirty...
        if v.datum_type().is_opaque() && v.volume() == 1 {
            if let Some(bqv) = v.as_slice::<Opaque>()?[0].downcast_ref::<BlockQuantValue>() {
                let opaque = Box::new(bqv.fact.clone());
                fact.opaque_fact = Some(opaque.clone());
                return self
                    .add_node(
                        name,
                        crate::ops::konst::Const::new_with_opaque_fact(v, opaque)?,
                        tvec!(fact),
                    )
                    .map(|id| id.into());
            }
        }
        self.add_node(name, crate::ops::konst::Const::new(v)?, tvec!(fact)).map(|id| id.into())
    }
}

impl TypedModel {
    pub fn into_optimized(mut self) -> TractResult<TypedModel> {
        self.declutter()?;
        self.optimize()?;
        Ok(self)
    }
    #[cfg(not(all(debug_assertions, feature = "paranoid_assertions")))]
    #[inline]
    pub fn check_consistency(&self) -> TractResult<()> {
        Ok(())
    }

    #[cfg(all(debug_assertions, feature = "paranoid_assertions"))]
    pub fn check_consistency(&self) -> TractResult<()> {
        self.check_edges()?;
        for node_id in &self.eval_order()? {
            let input_facts = self.node_input_facts(*node_id)?;
            let node = &self.nodes[*node_id];
            if node.id != *node_id {
                bail!("Node at position {} has id {}", node_id, node.id);
            }
            let output_facts = node.op.output_facts(&input_facts)?;
            if node.outputs.len() != output_facts.len() {
                bail!(
                    "Inconsistent model, node output count mismatch. Op says {}, node says {}. {}",
                    output_facts.len(),
                    node.outputs.len(),
                    node
                );
            }
            if node
                .outputs
                .iter()
                .map(|o| &o.fact)
                .zip(output_facts.iter())
                .any(|(a, b)| a.datum_type != b.datum_type || a.shape != b.shape)
            {
                bail!(
                    "Inconsistent model, output types mismatch. Op says: {:?}, node says: {:?}. {} with inputs {:?}. {}",
                    output_facts, node.outputs.iter().map(|o| &o.fact).collect::<Vec<_>>(), node, input_facts, node)
            }
            /* this is not true for regularly packed values
            if let Some(k) = node.op_as::<Const>() {
                ensure!(
                    !k.0.datum_type().is_opaque() || k.1.is_some(),
                    "Node {node} is missing an opaque fact"
                );
            }
            */
        }
        for node in &self.nodes {
            for (ix, output) in node.outputs.iter().enumerate() {
                output.fact.consistent().with_context(|| {
                    format!("Inconsistent fact {:?}: {:?}", OutletId::new(node.id, ix), output.fact)
                })?
            }
        }
        self.axes_mapping().context("Checking model axes mapping")?;
        Ok(())
    }

    pub fn into_decluttered(mut self) -> TractResult<TypedModel> {
        self.declutter()?;
        Ok(self)
    }

    /// Perform declutter passes on the network.
    pub fn transform(&mut self, transform: &dyn ModelTransform) -> TractResult<()> {
        transform.transform(self)
    }

    /// Perform declutter passes on the network.
    pub fn declutter(&mut self) -> TractResult<()> {
        crate::optim::Optimizer::declutter().session().optimize(self)
    }

    /// Perform optimization passes on the model, using a given optimizer session.
    pub fn optimize_with_session(&mut self, session: &mut OptimizerSession) -> TractResult<()> {
        session.optimize(self)?;
        self.properties.insert("tract_stage".to_string(), rctensor0("optimized".to_string()));
        Ok(())
    }

    pub fn concretize_dims(&self, values: &SymbolValues) -> TractResult<TypedModel> {
        values.translate_model(self)
    }

    pub fn prop_consts(&mut self) -> TractResult<()> {
        crate::optim::Optimizer::prop_consts().optimize(self)
    }

    /// Translate the graph to locally optimized operators (LIR or MIR ops).
    pub fn optimize(&mut self) -> TractResult<()> {
        crate::optim::Optimizer::codegen().optimize(self)
    }

    pub fn node_axes_mapping(&self, id: usize) -> TractResult<AxesMapping> {
        let (inputs, outputs) = self.node_facts(id)?;
        self.nodes[id].op.axes_mapping(&inputs, &outputs)
    }

    pub fn axes_mapping(&self) -> TractResult<AxesMapping> {
        crate::axes::for_model(self)
    }

    pub fn compute_const_facts(&mut self) -> TractResult<()> {
        for n in self.eval_order()? {
            let node = self.node(n);
            let (inputs, outputs) = self.node_facts(n)?;
            if node.op.is_stateless()
                && inputs.iter().all(|i| i.konst.is_some())
                && outputs.iter().any(|o| o.konst.is_none())
            {
                let inputs_ref =
                    inputs.iter().map(|f| f.konst.clone().unwrap().into_tvalue()).collect();
                match node.op.eval_with_session(&SessionState::default(), inputs_ref) {
                    Ok(res) => {
                        drop(inputs);
                        drop(outputs);
                        for (ix, output) in res.into_iter().enumerate() {
                            self.nodes[n].outputs[ix].fact.konst = Some(output.into_arc_tensor());
                        }
                    }
                    Err(e) => {
                        if !e.root_cause().is::<TooEarly>() {
                            Err(e).with_context(|| {
                                format!("Eager eval {} during const fact computation", self.node(n))
                            })?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

use crate::model::translator::Translate;
impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for SymbolValues {
    fn translate_node(
        &self,
        source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        target.check_consistency()?;
        let outlets = node.op.concretize_dims(source, node, target, mapping, self)?;
        for &outlet in &outlets {
            let fact = &mut target.nodes[outlet.node].outputs[outlet.slot].fact;
            if fact.shape.volume().is_zero() {
                if let Some(shape) = fact.shape.as_concrete() {
                    let tensor = Tensor::zero_dt(fact.datum_type, shape)?;
                    fact.konst = Some(tensor.into_arc_tensor());
                }
            }
            fact.consistent()?;
        }
        Ok(outlets)
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

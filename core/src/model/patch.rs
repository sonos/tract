use std::fmt::{Debug, Display};
use std::ops::{Deref, DerefMut};

use crate::hir::source::Source;
use crate::internal::*;
use crate::model::dsl::ModelSpecialOps;
use crate::model::*;
use crate::ops::source::TypedSource;

/// A change to apply to a model.
///
/// Actually structured around a model that represent the new nodes to be
/// inserted, plus information about how to connect these new nodes to the
/// pre-existing graph.
#[derive(Clone, Debug)]
pub struct ModelPatch<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    ModelImpl<TI, O>: ModelSpecialOps<TI, O>,
{
    /// the model-like 'pagch' of nodes to add to the model
    pub model: ModelImpl<TI, O>,
    pub incoming: HashMap<OutletId, OutletId>,
    pub shunt_outlet_by: HashMap<OutletId, OutletId>,
    pub obliterate: Vec<usize>,
}

impl<TI, O> Default for ModelPatch<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    ModelImpl<TI, O>: ModelSpecialOps<TI, O>,
{
    fn default() -> ModelPatch<TI, O> {
        ModelPatch {
            model: ModelImpl::default(),
            incoming: HashMap::new(),
            shunt_outlet_by: HashMap::new(),
            obliterate: vec![],
        }
    }
}

impl<TI, O> Deref for ModelPatch<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    ModelImpl<TI, O>: ModelSpecialOps<TI, O>,
{
    type Target = ModelImpl<TI, O>;
    fn deref(&self) -> &ModelImpl<TI, O> {
        &self.model
    }
}

impl<TI, O> DerefMut for ModelPatch<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    ModelImpl<TI, O>: ModelSpecialOps<TI, O>,
{
    fn deref_mut(&mut self) -> &mut ModelImpl<TI, O> {
        &mut self.model
    }
}

impl<TI, O> ModelPatch<TI, O>
where
    TI: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    ModelImpl<TI, O>: ModelSpecialOps<TI, O>,
{
    pub fn is_empty(&self) -> bool {
        self.model.nodes.is_empty() && self.shunt_outlet_by.is_empty() && self.obliterate.is_empty()
    }

    /// Draw a tap from a preexisting node.
    ///
    /// returns an OutletId usable in the little "patch" model
    pub fn tap_model(
        &mut self,
        model: &ModelImpl<TI, O>,
        outlet: OutletId,
    ) -> TractResult<OutletId> {
        let fact = model.outlet_fact(outlet)?;
        let id = self.add_source(
            format!("incoming-{}/{}", outlet.node, outlet.slot),
            dyn_clone::clone(fact),
        )?;
        self.incoming.insert(id, outlet);
        Ok(id)
    }

    /// Replace an Outlet in the target model by one from the patch.
    pub fn shunt_outside(&mut self, outlet: OutletId, by: OutletId) -> TractResult<()> {
        self.shunt_outlet_by.insert(outlet, by);
        Ok(())
    }

    pub fn obliterate(&mut self, node: usize) -> TractResult<()> {
        self.obliterate.push(node);
        Ok(())
    }

    /// Convenience method creating a patch that replace a single operation.
    pub fn replace_single_op<IO: Into<O>>(
        patched_model: &ModelImpl<TI, O>,
        node: &BaseNode<TI, O>,
        inputs: &[OutletId],
        new_op: IO,
    ) -> TractResult<ModelPatch<TI, O>> {
        let mut patch = ModelPatch::default();
        let new_op = new_op.into();
        let outputs = node.outputs.iter().map(|o| dyn_clone::clone(&o.fact)).collect();
        let by = patch.add_node(&*node.name, new_op, outputs)?;
        for (ix, i) in inputs.iter().enumerate() {
            let o = patch.tap_model(&patched_model, *i)?;
            patch.add_edge(o, InletId::new(by, ix))?;
        }
        for ix in 0..node.outputs.len() {
            patch.shunt_outside(OutletId::new(node.id, ix), OutletId::new(by, ix))?;
        }
        Ok(patch)
    }

    /// Convenience method creating a patch that replace a single operation.
    pub fn fuse_with_next<IO: Into<O>>(
        patched_model: &ModelImpl<TI, O>,
        node: &BaseNode<TI, O>,
        new_op: IO,
    ) -> TractResult<ModelPatch<TI, O>> {
        let mut patch = ModelPatch::default();
        let succ = if let Some(succ) = patched_model.single_succ(node.id)? {
            succ
        } else {
            bail!("Non single successor fuse attempt")
        };
        let new_op = new_op.into();
        let by = patch.add_node(&*node.name, new_op, tvec!(succ.outputs[0].fact.clone()))?;
        for (ix, i) in node.inputs.iter().enumerate() {
            let o = patch.tap_model(&patched_model, *i)?;
            patch.add_edge(o, InletId::new(by, ix))?;
        }
        for ix in 0..node.outputs.len() {
            patch.shunt_outside(OutletId::new(succ.id, ix), OutletId::new(by, ix))?;
        }
        Ok(patch)
    }

    /// Convenience method creating a patch that shunt the given node.
    pub fn shunt_one_op(
        patched_model: &ModelImpl<TI, O>,
        node: &BaseNode<TI, O>,
    ) -> TractResult<ModelPatch<TI, O>> {
        let mut patch = ModelPatch::default();
        let tap = patch.tap_model(patched_model, node.inputs[0])?;
        patch.shunt_outside(OutletId::new(node.id, 0), tap)?;
        Ok(patch)
    }

    /// Convenience method creating a patch that replace a single unary operation.
    pub fn single_unary_op<IO: Into<O>>(
        patched_model: &ModelImpl<TI, O>,
        node: &BaseNode<TI, O>,
        new_op: IO,
    ) -> TractResult<ModelPatch<TI, O>> {
        Self::replace_single_op(patched_model, node, &[node.inputs[0]], new_op)
    }

    /// Convenience method creating a patch that insert an unary op on an outlet.
    pub fn intercept<IO: Into<O>>(
        patched_model: &ModelImpl<TI, O>,
        outlet: OutletId,
        name: impl Into<String>,
        new_op: IO,
        fact: TI,
    ) -> TractResult<ModelPatch<TI, O>> {
        let mut patch = ModelPatch::default();
        let tap = patch.tap_model(patched_model, outlet)?;
        let new_id = patch.add_node(name, new_op, tvec!(fact))?;
        patch.add_edge(tap, InletId::new(new_id, 0))?;
        patch.shunt_outside(outlet, OutletId::new(new_id, 0))?;
        Ok(patch)
    }

    /// Apply all changes in the patch to the target model.
    pub fn apply(self, target: &mut ModelImpl<TI, O>) -> TractResult<()> {
        let prior_target_inputs = target.input_outlets()?.len();
        let prior_target_outputs = target.output_outlets()?.len();
        let ModelPatch { model: patch, incoming: mut mapping, shunt_outlet_by, obliterate } = self;
        let mut all_inputs = HashMap::new(); // new_id -> [ old_inputs ]
        for node in patch.nodes {
            if node.op_is::<Source>() || node.op_is::<TypedSource>() {
                continue;
            }
            let BaseNode { id, name, inputs, op, outputs } = node;
            let n_outputs = outputs.len();
            let facts = outputs.into_iter().map(|of| of.fact).collect();
            let added_node_id = target.add_node(name, op, facts)?;
            for ix in 0..n_outputs {
                mapping.insert(OutletId::new(id, ix), OutletId::new(added_node_id, ix));
            }
            all_inputs.insert(added_node_id, inputs);
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        for (outlet, by) in shunt_outlet_by {
            let fixed_by = mapping[&by];
            let succs = target.nodes()[outlet.node].outputs[outlet.slot].successors.clone();
            for succ in succs {
                target.add_edge(fixed_by, succ)?;
            }
            for o in target.outputs.iter_mut() {
                if *o == outlet {
                    *o = fixed_by;
                }
            }
            if let Some(label) = target.outlet_label(outlet).map(|s| s.to_string()) {
                target.set_outlet_label(fixed_by, label);
            }
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        for (node, inputs) in all_inputs {
            for (ix, input) in inputs.into_iter().enumerate() {
                target.add_edge(mapping[&input], InletId::new(node, ix))?;
            }
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        for node in obliterate {
            target.node_mut(node).op = target.create_dummy();
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        Ok(())
    }
}

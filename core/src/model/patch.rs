use crate::internal::*;
use crate::model::*;
use std::ops::{Deref, DerefMut};
use std::fmt::{ Display, Debug };

/// A change to apply to a model.
///
/// Actually structured around a model that represent the new nodes to be
/// inserted, plus information about how to connect these new nodes to the
/// pre-existing graph.
#[derive(Clone, Debug)]
pub struct ModelPatch<TI, O>
where TI: TensorInfo + Clone + 'static,
      O: Display + Debug + AsRef<Op> + AsMut<Op> + Clone + 'static
{
    /// the model-like 'patch' of nodes to add to the model
    pub model: ModelImpl<TI, O>,
    incoming: HashMap<OutletId, OutletId>,
    shunt_outlet_by: HashMap<OutletId, OutletId>,
}

impl<TI, O> Default for ModelPatch<TI, O>
where TI: TensorInfo + Clone + 'static,
      O: Display + Debug + AsRef<Op> + AsMut<Op> + Clone + 'static
{
    fn default() -> ModelPatch<TI, O> {
        ModelPatch {
            model: ModelImpl::default(),
            incoming: HashMap::new(),
            shunt_outlet_by: HashMap::new(),
        }
    }
}

impl<TI, O> Deref for ModelPatch<TI, O>
where TI: TensorInfo + Clone + 'static,
      O: Display + Debug + AsRef<Op> + AsMut<Op> + Clone + 'static
{
    type Target = ModelImpl<TI, O>;
    fn deref(&self) -> &ModelImpl<TI, O> {
        &self.model
    }
}

impl<TI, O> DerefMut for ModelPatch<TI, O>
where TI: TensorInfo + Clone + 'static,
      O: Display + Debug + AsRef<Op> + AsMut<Op> + Clone + 'static
{
    fn deref_mut(&mut self) -> &mut ModelImpl<TI, O> {
        &mut self.model
    }
}

impl<TI, O> ModelPatch<TI, O>
where TI: TensorInfo + Clone + 'static,
      O: Display + Debug + From<crate::ops::source::Source> + AsRef<Op> + AsMut<Op> + Clone + 'static
{
    /// Draw a tap from a preexisting node.
    ///
    /// returns an OutletId usable in the little "patch" model
    pub fn tap_model(&mut self, model: &ModelImpl<TI, O>, outlet: OutletId) -> TractResult<OutletId> {
        let fact = model.outlet_fact(outlet)?;
        let node_id = self
            .add_source(format!("incoming-{}/{}", outlet.node, outlet.slot), objekt::clone(fact))?;
        let inside = OutletId::new(node_id, 0);
        self.incoming.insert(inside, outlet);
        Ok(inside)
    }

    /// Replace an Outlet in the target model by one from the patch.
    pub fn shunt_outside(&mut self, outlet: OutletId, by: OutletId) -> TractResult<()> {
        self.shunt_outlet_by.insert(outlet, by);
        Ok(())
    }

    /// Convenience method creating a patch that replace a single operation.
    pub fn replace_single_op<IO: Into<O>>(
        patched_model: &ModelImpl<TI, O>,
        node: &Node<TI>,
        inputs: &[OutletId],
        new_op: IO,
    ) -> TractResult<ModelPatch<TI,O>> {
        let mut patch = ModelPatch::default();
        let new_op = new_op.into();
        let outputs = node.outputs.iter().map(|o| objekt::clone(&o.fact)).collect();
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

    /// Convenience method creating a patch that replace a single unary operation.
    pub fn single_unary_op<IO: Into<O>>(
        patched_model: &ModelImpl<TI, O>,
        node: &Node<TI>,
        new_op: IO,
    ) -> TractResult<ModelPatch<TI, O>> {
        Self::replace_single_op(patched_model, node, &[node.inputs[0]], new_op)
    }

    /// Apply all changes in the patch to the target model.
    pub fn apply(self, target: &mut ModelImpl<TI, O>) -> TractResult<()> {
        let ModelPatch { model: patch, incoming: mut mapping, shunt_outlet_by } = self;
        let mut all_inputs = HashMap::new(); // new_id -> [ old_inputs ]
        for node in patch.nodes {
            if node.op_is::<crate::ops::source::Source>() {
                continue;
            }
            let BaseNode { id, name, inputs, control_inputs, op, outputs } = node;
            let n_outputs = outputs.len();
            let facts = outputs.into_iter().map(|of| of.fact).collect();
            let added_node_id = target.add_node_disable_output_guess(name, op, facts, true)?;
            for &prec in control_inputs.iter() {
                target.nodes[added_node_id].control_inputs.push(mapping[&OutletId::new(prec, 0)].node)
            }
            for ix in 0..n_outputs {
                mapping.insert(OutletId::new(id, ix), OutletId::new(added_node_id, ix));
            }
            all_inputs.insert(added_node_id, inputs);
        }
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
        }
        for (node, inputs) in all_inputs {
            for (ix, input) in inputs.into_iter().enumerate() {
                target.add_edge(mapping[&input], InletId::new(node, ix))?;
            }
        }
        Ok(())
    }
}

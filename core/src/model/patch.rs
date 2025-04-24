use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::ops::{Deref, DerefMut};

use tract_data::itertools::{izip, Itertools};

use crate::internal::*;
use crate::model::*;

/// A change to apply to a model.
///
/// Actually structured around a model that represent the new nodes to be
/// inserted, plus information about how to connect these new nodes to the
/// pre-existing graph.
#[derive(Clone, Debug)]
pub struct ModelPatch<F, O>
where
    F: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    /// patch label for auditing and debugging
    pub context: Vec<String>,
    /// optimizer will ignore this patch in node to node loop if it was already
    /// encountered
    pub dont_apply_twice: Option<String>,
    /// the model-like 'patch' of nodes to add to the model
    pub model: Graph<F, O>,
    /// map of replaced inputs (patch node id to model node id)
    pub inputs: HashMap<usize, usize>,
    /// map of patch inputs to model wires
    pub taps: HashMap<OutletId, OutletId>,
    /// map of old model wires to be replaced by wires from the patch
    pub shunts: HashMap<OutletId, OutletId>,
    /// operations to discard from the model
    pub obliterate: Vec<usize>,
}

impl<F, O> Default for ModelPatch<F, O>
where
    F: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn default() -> ModelPatch<F, O> {
        ModelPatch {
            context: vec![],
            dont_apply_twice: None,
            model: Graph::default(),
            inputs: HashMap::default(),
            taps: HashMap::new(),
            shunts: HashMap::new(),
            obliterate: vec![],
        }
    }
}

impl<F, O> Deref for ModelPatch<F, O>
where
    F: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    type Target = Graph<F, O>;
    fn deref(&self) -> &Graph<F, O> {
        &self.model
    }
}

impl<F, O> DerefMut for ModelPatch<F, O>
where
    F: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn deref_mut(&mut self) -> &mut Graph<F, O> {
        &mut self.model
    }
}

impl<F, O> ModelPatch<F, O>
where
    F: Fact + Clone + 'static,
    O: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Graph<F, O>: SpecialOps<F, O>,
{
    pub fn new(s: impl Into<String>) -> Self {
        Self::default().with_context(s)
    }

    pub fn push_context(&mut self, s: impl Into<String>) {
        self.context.push(s.into());
    }

    pub fn with_context(mut self, s: impl Into<String>) -> Self {
        self.context.push(s.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.model.nodes.is_empty() && self.shunts.is_empty() && self.obliterate.is_empty()
    }

    /// Draw a tap from a preexisting node.
    ///
    /// returns an OutletId usable in the little "patch" model
    pub fn tap_model(&mut self, model: &Graph<F, O>, outlet: OutletId) -> TractResult<OutletId> {
        let fact = model.outlet_fact(outlet)?;
        let id = self.add_source(
            format!("tap.{}-{}/{}", model.node(outlet.node).name, outlet.node, outlet.slot),
            dyn_clone::clone(fact),
        )?;
        self.taps.insert(id, outlet);
        Ok(id)
    }

    /// Draw taps from a preexisting node.
    ///
    /// returns an OutletId usable in the little "patch" model
    pub fn taps<'a>(
        &mut self,
        model: &Graph<F, O>,
        outlets: impl IntoIterator<Item = &'a OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        outlets.into_iter().map(|o| self.tap_model(model, *o)).collect::<TractResult<TVec<_>>>()
    }

    pub unsafe fn shunt_outside_unchecked(
        &mut self,
        outlet: OutletId,
        by: OutletId,
    ) -> TractResult<()> {
        self.shunts.insert(outlet, by);
        Ok(())
    }

    /// Replace an Outlet in the target model by one from the patch.
    pub fn shunt_outside(
        &mut self,
        model: &Graph<F, O>,
        outlet: OutletId,
        by: OutletId,
    ) -> TractResult<()> {
        let original_fact = model.outlet_fact(outlet)?;
        let new_fact = self.model.outlet_fact(by)?;
        if !original_fact.compatible_with(new_fact) {
            bail!(
                "Trying to substitute a {:?} by {:?} as output #{} of {}.\n{:?}",
                original_fact,
                new_fact,
                outlet.slot,
                model.node(outlet.node),
                self
            );
        }
        self.shunts.insert(outlet, by);
        Ok(())
    }

    pub fn obliterate(&mut self, node: usize) -> TractResult<()> {
        self.obliterate.push(node);
        Ok(())
    }

    /// Convenience method creating a patch that replaces a single operation.
    pub fn replace_single_op<IO: Into<O>>(
        patched_model: &Graph<F, O>,
        node: &Node<F, O>,
        inputs: &[OutletId],
        new_op: IO,
    ) -> TractResult<ModelPatch<F, O>> {
        let mut patch = ModelPatch::default();
        let new_op = new_op.into();
        let inputs = patch.taps(patched_model, inputs)?;
        let wires = patch.wire_node(&node.name, new_op, &inputs)?;
        for (ix, o) in wires.iter().enumerate() {
            patch.shunt_outside(patched_model, OutletId::new(node.id, ix), *o)?;
        }
        patch.obliterate(node.id)?;
        Ok(patch)
    }

    /// Convenience method creating a patch that fuses an op with the next one.
    pub fn fuse_with_next<IO: Into<O>>(
        patched_model: &Graph<F, O>,
        node: &Node<F, O>,
        new_op: IO,
    ) -> TractResult<ModelPatch<F, O>> {
        let mut patch = ModelPatch::default();
        let succ = if let Some(succ) = patched_model.single_succ(node.id)? {
            succ
        } else {
            bail!("Non single successor fuse attempt")
        };
        let inputs = patch.taps(patched_model, &node.inputs)?;
        let output = patch.wire_node(&node.name, new_op.into(), &inputs)?;
        patch.shunt_outside(patched_model, succ.id.into(), output[0])?;
        Ok(patch)
    }

    /// Convenience method creating a patch that shunts the given node.
    pub fn shunt_one_op(
        patched_model: &Graph<F, O>,
        node: &Node<F, O>,
    ) -> TractResult<Option<ModelPatch<F, O>>> {
        ensure!(node.inputs.len() == 1);
        ensure!(node.outputs.len() == 1);
        if patched_model.outputs.contains(&node.id.into())
            && patched_model.outputs.contains(&node.inputs[0])
        {
            Ok(None)
        } else {
            Self::rewire(patched_model, &node.inputs, &[node.id.into()], &|_p, xs| Ok(xs.into()))
                .with_context(|| format!("Shunting {node}"))
                .map(Some)
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn rewire(
        patched_model: &Graph<F, O>,
        from: &[OutletId],
        to: &[OutletId],
        wiring: &dyn Fn(&mut Self, &[OutletId]) -> TractResult<TVec<OutletId>>,
    ) -> TractResult<ModelPatch<F, O>> {
        let mut patch = ModelPatch::default();
        let taps = patch.taps(patched_model, from)?;
        let news = wiring(&mut patch, &taps)?;
        if news.len() != to.len() {
            bail!(
                "Wrong number of outputs for rewiring, expected {}, function returned {}",
                to.len(),
                news.len()
            );
        }
        for (new, &old) in izip!(news, to) {
            patch.shunt_outside(patched_model, old, new)?;
        }
        Ok(patch)
    }

    /// Convenience method creating a patch that replace a single unary operation.
    pub fn single_unary_op<IO: Into<O>>(
        patched_model: &Graph<F, O>,
        node: &Node<F, O>,
        new_op: IO,
    ) -> TractResult<ModelPatch<F, O>> {
        Self::replace_single_op(patched_model, node, &[node.inputs[0]], new_op)
    }

    /// Convenience method creating a patch that insert an unary op on an outlet.
    pub fn intercept<IO: Into<O>>(
        patched_model: &Graph<F, O>,
        outlet: OutletId,
        name: impl Into<String>,
        new_op: IO,
        fact: F,
    ) -> TractResult<ModelPatch<F, O>> {
        let mut patch = ModelPatch::default();
        let tap = patch.tap_model(patched_model, outlet)?;
        let new_id = patch.add_node(name, new_op, tvec!(fact))?;
        patch.add_edge(tap, InletId::new(new_id, 0))?;
        patch.shunt_outside(patched_model, outlet, OutletId::new(new_id, 0))?;
        Ok(patch)
    }

    pub fn wire_node(
        &mut self,
        name: impl Into<String>,
        op: impl Into<O>,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut name = name.into();
        if self.nodes.iter().any(|n| n.name == *name) {
            for i in 1.. {
                let s = format!("{name}#{i}");
                if self.nodes.iter().all(|n| n.name != s) {
                    name = s;
                    break;
                }
            }
        }
        self.model.wire_node(name, op.into(), inputs)
    }

    /// Apply all changes in the patch to the target model.
    pub fn apply(self, target: &mut Graph<F, O>) -> TractResult<()> {
        let prior_target_inputs = target.input_outlets()?.len();
        let prior_target_outputs = target.output_outlets()?.len();
        let ModelPatch {
            model: patch,
            taps: mut mapping,
            shunts: shunt_outlet_by,
            obliterate,
            inputs: replaced_inputs,
            ..
        } = self;
        let mut all_inputs = HashMap::new(); // new_node_id_in_model -> [ patch_outlet_id ]
        let mut model_input_outlets = target.input_outlets()?.to_vec();
        let mut new_nodes = HashSet::new();
        for node in patch.nodes {
            if <Graph<F, O>>::is_source(&node.op)
                && mapping.contains_key(&OutletId::new(node.id, 0))
            {
                // this is a tap
                continue;
            }
            let Node { id: patch_node_id, name, inputs, op, outputs } = node;
            let n_outputs = outputs.len();
            for dup in 0..target.nodes.len() {
                if target.node(dup).op().same_as(op.as_ref())
                    && inputs.len() == target.node(dup).inputs.len()
                    && inputs
                        .iter()
                        .zip(target.node(dup).inputs.iter())
                        .all(|(patch_input, d)| mapping[patch_input] == *d)
                {
                    for ix in 0..n_outputs {
                        mapping.insert(OutletId::new(patch_node_id, ix), OutletId::new(dup, ix));
                    }
                    continue;
                }
            }
            let facts = outputs.into_iter().map(|of| of.fact).collect();
            let added_node_id = target.add_node(name, op, facts)?;
            new_nodes.insert(added_node_id);
            for ix in 0..n_outputs {
                mapping.insert(OutletId::new(patch_node_id, ix), OutletId::new(added_node_id, ix));
            }
            all_inputs.insert(added_node_id, inputs);
            if <Graph<F, O>>::is_source(&target.node(added_node_id).op) {
                // this is actually an input replacement
                model_input_outlets.iter_mut().for_each(|oo| {
                    if oo.node == replaced_inputs[&patch_node_id] {
                        oo.node = added_node_id;
                    }
                });
            }
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        for (&outlet, &by) in shunt_outlet_by.iter().sorted() {
            let replace_by = mapping[&by];
            let succs = target.nodes()[outlet.node].outputs[outlet.slot].successors.clone();
            for succ in succs {
                target.add_edge(replace_by, succ)?;
            }
            for o in target.outputs.iter_mut() {
                if *o == outlet {
                    *o = replace_by;
                }
            }
            if let Some(label) = target.outlet_labels.remove(&outlet) {
                target.set_outlet_label(replace_by, label)?;
            }
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        for (&node, inputs) in all_inputs.iter().sorted() {
            for (ix, input) in inputs.iter().enumerate() {
                target.add_edge(mapping[input], InletId::new(node, ix))?;
            }
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        for node in obliterate {
            target.node_mut(node).op = target.create_dummy();
        }
        debug_assert_eq!(target.input_outlets()?.len(), prior_target_inputs);
        debug_assert_eq!(target.output_outlets()?.len(), prior_target_outputs);
        target.set_input_outlets(&model_input_outlets)?;
        let mut maybe_garbage: HashSet<usize> = shunt_outlet_by.iter().map(|o| o.0.node).collect();
        while let Some(&maybe) = maybe_garbage.iter().next() {
            maybe_garbage.remove(&maybe);
            if !target.outputs.iter().any(|output| output.node == maybe)
                && !target.inputs.iter().any(|input| input.node == maybe)
                && target.node(maybe).outputs.iter().all(|of| of.successors.is_empty())
            {
                target.node_mut(maybe).op = target.create_dummy();
                target.node_mut(maybe).name = format!("Dummy-node-{maybe}");
                target.node_mut(maybe).outputs.clear(); // necessary to drop facts and consts
                let inputs = std::mem::take(&mut target.node_mut(maybe).inputs);
                for &i in &inputs {
                    target.node_mut(i.node).outputs[i.slot].successors.retain(|s| s.node != maybe);
                    maybe_garbage.insert(i.node);
                }
                target.check_edges()?;
            }
        }
        for n in new_nodes.iter() {
            if let Some((prefix, _)) = target.nodes[*n].name.split_once('#') {
                target.nodes[*n].name = target.unique_name(prefix).into();
            } else if target
                .nodes
                .iter()
                .any(|node| node.id != *n && target.nodes[*n].name == node.name)
            {
                target.nodes[*n].name = target.unique_name(&target.nodes[*n].name).to_string();
            }
        }
        Ok(())
    }
}

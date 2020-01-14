use crate::model::{Fact, InletId, ModelImpl, OutletId};
use crate::prelude::*;
use std::collections::HashMap;
use std::fmt;

trait GraphRewriter<F, O>
where
    F: Fact + Clone + 'static,
    O: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    fn rewrite_fact(
        &self,
        _old: &ModelImpl<F, O>,
        new: &ModelImpl<F, O>,
        map: &HashMap<OutletId, OutletId>,
        outlet: OutletId,
    ) -> TractResult<F> {
        Ok(new.outlet_fact(map[&outlet])?.clone())
    }

    fn rewrite_op(
        &self,
        _old: &ModelImpl<F, O>,
        _new: &ModelImpl<F, O>,
        _map: &HashMap<OutletId, OutletId>,
        node: &BaseNode<F, O>,
    ) -> TractResult<O> {
        Ok(node.op.clone())
    }

    fn rewrite_model(&self, old: &ModelImpl<F, O>) -> TractResult<ModelImpl<F, O>> {
        let mut new = ModelImpl::default();
        let mut map = HashMap::new();
        for old_id in old.eval_order()? {
            let old_node = &old.nodes()[old_id];
            let facts = old_node
                .outputs
                .iter()
                .enumerate()
                .map(|(ix, _)| self.rewrite_fact(old, &new, &map, OutletId::new(old_id, ix)))
                .collect::<TractResult<TVec<_>>>()?;
            let new_id = new.add_node(
                old_node.name.clone(),
                self.rewrite_op(old, &new, &map, old_node)?,
                facts,
            )?;
            for ix in 0..old_node.outputs.len() {
                map.insert(OutletId::new(old_id, ix), OutletId::new(new_id, ix));
            }
            for ix in 0..old_node.outputs.len() {
                if let Some(label) = old.outlet_label(OutletId::new(old_id, ix)) {
                    new.set_outlet_label(OutletId::new(new_id, ix), label.to_string());
                }
            }
            if old.input_outlets()?.contains(&OutletId::new(old_node.id, 0)) {
                continue;
            }
            for (ix, input) in old_node.inputs.iter().enumerate() {
                new.add_edge(map[&input], InletId::new(new_id, ix))?;
            }
            for input in old_node.control_inputs.iter() {
                new.node_mut(new_id).control_inputs.push(map[&OutletId::new(*input, 0)].node);
            }
        }
        for i in old.input_outlets()? {
            if !map.contains_key(&i) {
                let node = old.node(i.node);
                debug!("Translate useless source {}", node);
                let new_id = new.add_node(
                    &*node.name,
                    node.op.clone(),
                    tvec!(node.outputs[0].fact.clone()),
                )?;
                map.insert(*i, new_id.into());
            }
        }
        // maintaining order of i/o interface
        new.inputs = old.input_outlets()?.iter().map(|i| map[&i]).collect();
        new.outputs = old.output_outlets()?.iter().map(|o| map[&o]).collect();
        Ok(new)
    }
}

struct NoopRewriter;
impl<F, O> GraphRewriter<F, O> for NoopRewriter
where
    F: Fact + Clone + 'static,
    O: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
}

pub(crate) fn compact<F, O>(old: &ModelImpl<F, O>) -> TractResult<ModelImpl<F, O>>
where
    F: Fact + Clone + 'static,
    O: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    NoopRewriter.rewrite_model(old)
}

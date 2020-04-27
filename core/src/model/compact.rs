use crate::model::{Fact, ModelImpl, OutletId};
use crate::internal::*;
use std::collections::HashMap;
use std::fmt;

trait GraphRewriter<F, O>
where
    F: Fact + Clone + 'static + Hash,
    O: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
{
    fn wire_node(
        &self,
        old: &ModelImpl<F, O>,
        new: &mut ModelImpl<F, O>,
        map: &HashMap<OutletId, OutletId>,
        node: &BaseNode<F, O>,
    ) -> TractResult<TVec<OutletId>>;

    fn rewrite_model(&self, old: &ModelImpl<F, O>) -> TractResult<ModelImpl<F, O>> {
        let mut new = ModelImpl::default();
        let mut map = HashMap::new();
        for old_id in old.eval_order()? {
            let old_node = old.node(old_id);
            let outlets = self.wire_node(old, &mut new, &map, old_node)?;
            for (ix, &o) in outlets.iter().enumerate() {
                map.insert(OutletId::new(old_id, ix), o);
                if let Some(label) = old.outlet_label(OutletId::new(old_id, ix)) {
                    new.set_outlet_label(o, label.to_string());
                }
            }
            if old.input_outlets()?.contains(&OutletId::new(old_node.id, 0)) {
                continue;
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
    F: Fact + Clone + 'static + Hash,
    O: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    ModelImpl<F, O>: ModelWireNode<F, O> + ModelSpecialOps<F, O>,
{
    fn wire_node(
        &self,
        old: &ModelImpl<F, O>,
        new: &mut ModelImpl<F, O>,
        map: &HashMap<OutletId, OutletId>,
        node: &BaseNode<F, O>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|o| map[o]).collect::<TVec<_>>();
        let outlets = new
            .wire_node(&*node.name, node.op.clone(), &inputs)
            .chain_err(|| format!("Compacting model, {}", node))?;
        for (ix, &o) in outlets.iter().enumerate() {
            new.set_outlet_fact(o, old.outlet_fact(OutletId::new(node.id, ix))?.clone())?;
        }
        Ok(outlets)
    }
}

pub fn compact<F, O>(old: &ModelImpl<F, O>) -> TractResult<ModelImpl<F, O>>
where
    F: Fact + Clone + 'static + Hash,
    O: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Hash,
    ModelImpl<F, O>: ModelWireNode<F, O> + ModelSpecialOps<F, O>,
{
    NoopRewriter.rewrite_model(old)
}

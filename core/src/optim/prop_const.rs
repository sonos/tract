use crate::model::order::eval_order_for_nodes;
use crate::model::*;
use crate::TractResult;
use bit_set;

#[derive(Debug)]
pub struct PropConst;

impl super::TypedPass for PropConst {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut replaced = 0;
        let mut done = bit_set::BitSet::with_capacity(model.nodes().len());
        let mut needed: Vec<usize> = vec![];
        for t in model.output_outlets()?.iter().map(|n| n.node) {
            needed.push(t);
        }
        while let Some(&node) = needed.last() {
            if done.contains(node) {
                needed.pop();
                continue;
            }
            if model.nodes()[node].inputs.iter().all(|i| done.contains(i.node)) {
                needed.pop();
                done.insert(node);
            } else {
                trace!("Looking at node {} inputs", model.nodes()[node]);
                for ix in 0..model.nodes()[node].inputs.len() {
                    let source = model.nodes()[node].inputs[ix];
                    if model.nodes()[source.node].op().name() != "Const"
                        && model.outlet_fact(source)?.konst.is_some()
                        && eval_order_for_nodes(
                            model.nodes(),
                            &model.input_outlets()?.iter().map(|n| n.node).collect::<Vec<_>>(),
                            &[source.node],
                            &[]
                        )?
                        .into_iter()
                        .all(|n| model.nodes()[n].op().as_stateless().is_some())
                    {
                        let konst = model.outlet_fact(source)?.konst.clone().unwrap();
                        let id = model.nodes().len();
                        trace!(
                            "   Replacing node {} input {} by a constant instead of {:?}",
                            model.nodes()[node],
                            ix,
                            source
                        );
                        let id = model.add_const(format!("Const-{}", id), konst.clone())?;
                        model.add_edge(id, InletId::new(node, ix))?;
                        model.check_edges()?;
                        model.set_outlet_fact(id, konst.into())?;
                        replaced += 1;
                    } else {
                        needed.push(source.node);
                    }
                }
            }
        }
        debug!("Replaced {} inputs by constants", replaced);
        Ok(replaced > 0)
    }
}

use crate::model::*;
use crate::TractResult;
use bit_set;

#[derive(Debug)]
pub struct PropConst;

impl super::TypedPass for PropConst {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut propagated = 0;
        let mut stateful = bit_set::BitSet::with_capacity(model.nodes().len());
        for node in model.eval_order()? {
            if model.node(node).op.as_stateless().is_none()
                || model.node(node).inputs.iter().any(|i| stateful.contains(i.node))
            {
                stateful.insert(node);
                continue;
            }
            if model.node_input_facts(node)?.iter().all(|i| i.konst.is_some())
                && model.node_output_facts(node)?.iter().any(|i| i.konst.is_none())
            {
                let inputs = model
                    .node_input_facts(node)?
                    .iter()
                    .map(|i| i.konst.clone().unwrap())
                    .collect();
                let outputs = model.node(node).op.as_stateless().unwrap().eval(inputs)?;
                for (ix, value) in outputs.into_iter().enumerate() {
                    model.node_mut(node).outputs[ix].fact.konst = Some(value);
                    propagated += 1;
                }
            }
        }
        debug!("propagated {} consts", propagated);
        let mut replaced = 0;
        let mut patch = TypedModelPatch::default();
        for node in model.eval_order()? {
            if model.node_input_facts(node)?.iter().any(|f| f.konst.is_some())
                && model.node_input_facts(node)?.iter().any(|f| f.konst.is_none())
            {
                debug!("Cleanup inputs for {}", model.node(node));
                for i in 0..model.node(node).inputs.len() {
                    if let Some(k) = model.node_input_facts(node)?[i].konst.clone() {
                        let outlet = model.node(node).inputs[i];
                        let knode = model.node(outlet.node);
                        if stateful.contains(knode.id) || knode.op_is::<crate::ops::konst::Const>() {
                            continue;
                        }
                        let k = patch.add_const(&*knode.name, k)?;
                        patch.shunt_outside(model, outlet, k)?;
                        replaced += 1;
                    }
                }
            }
        }
        if replaced > 0 {
            patch.apply(model)?;
            debug!("replaced {} consts", propagated);
        }
        Ok(replaced + propagated > 0)
    }
}

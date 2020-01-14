use crate::model::{Fact, InletId, ModelImpl, OutletId};
use crate::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;

pub(crate) fn compact<TI1, TI2, O1, O2, E1, E2>(
    old: &ModelImpl<TI1, O1>,
) -> TractResult<ModelImpl<TI2, O2>>
where
    TractError: From<E1> + From<E2>,
    TI1: Fact + Clone + 'static,
    TI2: Fact + TryFrom<TI1, Error = E1> + Clone + 'static,
    O1: fmt::Display + fmt::Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: fmt::Display + TryFrom<O1, Error = E2> + fmt::Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let mut model = ModelImpl::default();
    let mut map = HashMap::new();
    for old_id in old.eval_order()? {
        let old_node = &old.nodes()[old_id];
        let facts = old_node
            .outputs
            .iter()
            .map(|of| Ok(TI2::try_from(of.fact.clone())?))
            .collect::<TractResult<TVec<_>>>()
            .map_err(|e| format!("While translating {}: {:?}", old_node, e))?;
        let new_op = O2::try_from(old_node.op.clone())?;
        let new_id = model.add_node(old_node.name.clone(), new_op, facts)?;
        map.insert(old_id, new_id);
        for ix in 0..old_node.outputs.len() {
            if let Some(label) = old.outlet_label(OutletId::new(old_id, ix)) {
                model.set_outlet_label(OutletId::new(new_id, ix), label.to_string());
            }
        }
        if old.input_outlets()?.contains(&OutletId::new(old_node.id, 0)) {
            continue;
        }
        for (ix, input) in old_node.inputs.iter().enumerate() {
            model
                .add_edge(OutletId::new(map[&input.node], input.slot), InletId::new(new_id, ix))?;
        }
        for input in old_node.control_inputs.iter() {
            model.node_mut(new_id).control_inputs.push(map[input]);
        }
    }
    for i in old.input_outlets()? {
        if !map.contains_key(&i.node) {
            let node = old.node(i.node);
            debug!("Translate useless source {}", node);
            let new_id = model.add_node(
                &*node.name,
                O2::try_from(node.op.clone())?,
                tvec!(TI2::try_from(node.outputs[0].fact.clone())?),
            )?;
            map.insert(i.node, new_id);
        }
    }
    // maintaining order of i/o interface
    model.inputs = old.input_outlets()?.iter().map(|i| OutletId::new(map[&i.node], 0)).collect();
    model.outputs =
        old.output_outlets()?.iter().map(|o| OutletId::new(map[&o.node], o.slot)).collect();
    Ok(model)
}

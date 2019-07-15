use crate::model::{InletId, ModelImpl, OutletId, TensorInfo};
use crate::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{ Display, Debug };

pub(crate) fn translate<TI1, TI2, O1, O2, E1, E2>(old: &ModelImpl<TI1, O1>) -> TractResult<ModelImpl<TI2, O2>>
where
    TractError: From<E1> + From<E2>,
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + TryFrom<TI1, Error=E1> + Clone + 'static,
    O1: Display + Debug + Clone + AsRef<Op> + AsMut<Op> + Clone + 'static,
    O2: Display + TryFrom<O1, Error=E2> + Debug + AsRef<Op> + AsMut<Op> + Clone + 'static,
{
    let mut model = ModelImpl::default();
    for old_node in old.nodes() {
        let facts = old_node
            .outputs
            .iter()
            .map(|of| Ok(TI2::try_from(of.fact.clone())?))
            .collect::<TractResult<TVec<_>>>()
            .map_err(|e| format!("While translating {}: {:?}", old_node, e))?;
        let new_op = O2::try_from(old_node.op.clone())?;
        model.add_node(old_node.name.clone(), new_op, facts)?;
        model.node_mut(old_node.id).outputs.iter_mut().zip(old_node.outputs.iter())
            .for_each(|(new, old)| new.successors = old.successors.clone());
        model.node_mut(old_node.id).inputs = old_node.inputs.clone();
        model.node_mut(old_node.id).control_inputs = old_node.control_inputs.clone();
    }
    model.inputs = old.input_outlets()?.to_vec();
    model.outputs = old.output_outlets()?.to_vec();
    Ok(model)
}

pub(crate) fn compact<TI1, TI2, O1, O2, E1, E2>(old: &ModelImpl<TI1, O1>) -> TractResult<ModelImpl<TI2, O2>>
where
    TractError: From<E1> + From<E2>,
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + TryFrom<TI1, Error=E1> + Clone + 'static,
    O1: Display + Debug + Clone + AsRef<Op> + AsMut<Op> + Clone + 'static,
    O2: Display + TryFrom<O1, Error=E2> + Debug + AsRef<Op> + AsMut<Op> + Clone + 'static,
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
    // maintaining order of i/o interface
    model.inputs = old
        .input_outlets()?
        .iter()
        .filter_map(|i| map.get(&i.node).map(|&n| OutletId::new(n, i.slot)))
        .collect();
    model.outputs =
        old.output_outlets()?.iter().map(|o| OutletId::new(map[&o.node], o.slot)).collect();
    Ok(model)
}

use crate::model::{InletId, ModelImpl, OutletId, TensorInfo};
use crate::ops::Translate;
use crate::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{Debug, Display};

pub(crate) fn translate<TI1, TI2, O1, O2, Ctx>(
    source: &ModelImpl<TI1, O1>,
    ctx: &Ctx,
) -> TractResult<(ModelImpl<TI2, O2>, HashMap<OutletId, OutletId>)>
where
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + Clone + 'static,
    O1: Display
        + Debug
        + AsRef<dyn Op>
        + AsMut<dyn Op>
        + Clone
        + 'static
        + Translate<TI1, O1, TI2, O2, Ctx>,
    O2: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let mut target = ModelImpl::default();
    let mut mapping = HashMap::new();
    for old_id in source.eval_order()? {
        let node = source.node(old_id);
        let outlets = node
            .op
            .translate(&source, node, &mut target, &mapping, ctx)
            .chain_err(|| format!("Translating {}", node))?;
        for (ix, outlet) in outlets.into_iter().enumerate() {
            mapping.insert(OutletId::new(node.id, ix), outlet);
        }
    }
    // maintaining order of i/o interface
    target.inputs = source.input_outlets()?.iter().map(|i| mapping[&i]).collect();
    target.outputs = source.output_outlets()?.iter().map(|o| mapping[&o]).collect();
    Ok((target, mapping))
}

pub(crate) fn compact<TI1, TI2, O1, O2, E1, E2>(
    old: &ModelImpl<TI1, O1>,
) -> TractResult<ModelImpl<TI2, O2>>
where
    TractError: From<E1> + From<E2>,
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + TryFrom<TI1, Error = E1> + Clone + 'static,
    O1: Display + Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: Display + TryFrom<O1, Error = E2> + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
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

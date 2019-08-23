use crate::model::{InletId, ModelImpl, OutletId, TensorInfo};
use crate::ops::Translate;
use crate::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt::{ Display, Debug };

pub(crate) fn translate2<TI1, TI2, O1, O2, E1, E2, Ctx>(source: &ModelImpl<TI1, O1>, ctx: &Ctx) -> TractResult<(ModelImpl<TI2, O2>, HashMap<OutletId, OutletId>)>
where
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + Clone + 'static,
    O1: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static + Translate<TI1, O1, TI2, O2, Ctx>,
    O2: Display + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
        let mut target = ModelImpl::default();
        let mut mapping = HashMap::new();
        for old_id in source.eval_order()? {
            trace!(
                "Translate node {} {} ({})",
                old_id,
                source.node(old_id).name,
                source.node(old_id).op().name()
            );
            let node = source.node(old_id);
            let outlets = node
                .op
                .translate(&source, node, &mut target, &mapping, ctx)
                .chain_err(|| format!("Translating {:?}", node))?;
            for (ix, outlet) in outlets.into_iter().enumerate() {
                mapping.insert(OutletId::new(node.id, ix), outlet);
            }
            trace!("Target is now {}", target.nodes().len());
        }
        // maintaining order of i/o interface
        target.inputs = source.input_outlets()?.iter().map(|i| mapping[&i]).collect();
        target.outputs = source.output_outlets()?.iter().map(|o| mapping[&o]).collect();
        Ok((target, mapping))
}

pub(crate) fn translate<TI1, TI2, O1, O2, E1, E2>(old: &ModelImpl<TI1, O1>) -> TractResult<ModelImpl<TI2, O2>>
where
    TractError: From<E1> + From<E2>,
    TI1: TensorInfo + Clone + 'static,
    TI2: TensorInfo + TryFrom<TI1, Error=E1> + Clone + 'static,
    O1: Display + Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: Display + TryFrom<O1, Error=E2> + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
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
    O1: Display + Debug + Clone + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    O2: Display + TryFrom<O1, Error=E2> + Debug + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
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

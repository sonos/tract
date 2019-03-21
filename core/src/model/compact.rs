use crate::datum::TryInto;
use crate::model::{InletId, Model, OutletId, TensorInfo};
use crate::{TVec, TractResult};
use std::collections::HashMap;

pub(crate) fn compact<TI1, TI2>(old: &Model<TI1>) -> TractResult<Model<TI2>>
where
    TI1: TensorInfo,
    TI2: TensorInfo,
    TI1: TryInto<TI2>,
{
    let mut model = Model::default();
    let mut map = HashMap::new();
    for old_id in old.eval_order()? {
        let old_node = &old.nodes()[old_id];
        let facts = old_node
            .outputs
            .iter()
            .map(|of| of.fact.try_into())
            .collect::<TractResult<TVec<_>>>()
            .map_err(|e| format!("While translating {}: {:?}", old_node, e))?;
        let new_id = model.add_node(old_node.name.clone(), old_node.op.clone(), facts)?;
        map.insert(old_id, new_id);
        if old.inputs()?.contains(&OutletId::new(old_node.id, 0)) {
            continue;
        }
        for (ix, input) in old_node.inputs.iter().enumerate() {
            model
                .add_edge(OutletId::new(map[&input.node], input.slot), InletId::new(new_id, ix))?;
        }
    }
    // maintaining order of i/o interface
    model.inputs = old.inputs()?.iter().map(|i| OutletId::new(map[&i.node], i.slot)).collect();
    model.outputs = old.outputs()?.iter().map(|o| OutletId::new(map[&o.node], o.slot)).collect();
    Ok(model)
}

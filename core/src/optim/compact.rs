use model::{InletId, OutletId};
use std::collections::HashMap;
use {Model, TractResult};

pub fn compact(old: &Model) -> TractResult<Model> {
    let mut model = Model::default();
    let mut map = HashMap::new();
    for old_id in old.eval_order()? {
        let old_node = &old.nodes()[old_id];
        let new_id = model.add_node(old_node.name.clone(), old_node.op.clone())?;
        map.insert(old_id, new_id);
        for (ix, output) in old_node.outputs.iter().enumerate() {
            model.set_fact(OutletId::new(new_id, ix), output.fact.clone())?;
        }
        if old.inputs()?.contains(&OutletId::new(old_node.id, 0)) {
            continue
        }
        for (ix, input) in old_node.inputs.iter().enumerate() {
            model.add_edge(
                OutletId::new(map[&input.node], input.slot),
                InletId::new(new_id, ix),
            )?;
        }
    }
    // maintaining order of i/o interface
    model.inputs = old
        .inputs()?
        .iter()
        .map(|i| OutletId::new(map[&i.node], i.slot))
        .collect();
    model.outputs = old
        .outputs()?
        .iter()
        .map(|o| OutletId::new(map[&o.node], o.slot))
        .collect();
    Ok(model)
}

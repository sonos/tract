use bit_set;
use { Model, TfdResult };
use model::{InletId, OutletId};

pub fn prop_const(model: &mut Model) -> TfdResult<()> {
    let mut done = bit_set::BitSet::with_capacity(model.nodes().len());
    let mut needed: Vec<usize> = vec![];
    for t in model.outputs()?.iter().map(|n| n.node) {
        needed.push(t);
    }
    while let Some(&node) = needed.last() {
        if done.contains(node) {
            needed.pop();
            continue;
        }
        if model.nodes()[node]
            .inputs
            .iter()
            .all(|i| done.contains(i.node))
        {
            needed.pop();
            done.insert(node);
        } else {
            for ix in 0..model.nodes()[node].inputs.len() {
                use analyser::types::Fact;
                let source = model.nodes()[node].inputs[ix];
                if model.nodes()[source.node].op().name() != "Const"
                    && model.fact(source)?.is_concrete()
                {
                    use model::ModelDsl;
                    let konst = model.fact(source)?.concretize().unwrap();
                    let id = model.nodes().len();
                    let id = model.add_const(format!("Const-{}", id), konst.clone())?;
                    model.add_edge(OutletId::new(id, 0), InletId::new(node, ix))?;
                    model.set_fact(OutletId::new(id, 0), konst.into())?;
                } else {
                    needed.push(source.node);
                }
            }
        }
    }
    Ok(())
}


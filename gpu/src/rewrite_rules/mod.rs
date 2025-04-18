use tract_core::model::{TypedModel, TypedNode};
use tract_core::prelude::TVec;

pub mod rewire_syncs;

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}

pub fn next_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
        return None;
    }
    let succ = node.outputs[0].successors[0];
    Some(&model.nodes()[succ.node])
}

pub fn previous_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.inputs.len() != 1 {
        return None;
    }
    Some(&model.nodes()[node.inputs[0].node])
}

pub fn previous_nodes<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a TypedNode> {
    node.inputs.iter().map(|n| &model.nodes()[n.node]).collect()
}

mod rewire_metal_sync;
mod rms_norm;
mod silu;

use tract_core::internal::*;
use tract_core::ops::konst::Const;

pub use rewire_metal_sync::rewire_metal_sync;
pub use rms_norm::{as_rms_norm_rule, BasicRmsNorm};
pub use silu::{as_silu_rule, BasicSilu};

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}

fn next_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
        return None;
    }
    let succ = node.outputs[0].successors[0];
    Some(&model.nodes()[succ.node])
}

fn previous_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.inputs.len() != 1 {
        return None;
    }
    Some(&model.nodes()[node.inputs[0].node])
}

fn collect_node_const_inputs<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a Const> {
    node.inputs
        .iter()
        .filter_map(|i| {
            let prec = &model.nodes()[i.node];
            prec.op_as::<Const>()
        })
        .collect::<TVec<_>>()
}

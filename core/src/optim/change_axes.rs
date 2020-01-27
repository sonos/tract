use super::TypedPass;
use crate::model::*;
use crate::ops::invariants::*;
use crate::TractResult;
use std::collections::HashMap;

pub fn change_axes(model: &mut TypedModel, change: &AxisChange, lock_interfaces: bool) -> TractResult<bool> {
    let mut todo_changes = vec![change.clone()];
    let mut changed_wires = HashMap::new();
    changed_wires.insert(change.outlet, change.op.clone());
    let mut changed_ops: HashMap<usize, Box<dyn TypedOp>> = HashMap::new();
    while let Some(change) = todo_changes.pop() {
        if lock_interfaces && (model.output_outlets()?.contains(&change.outlet)
            || model.input_outlets()?.contains(&change.outlet))
        {
            return Ok(false);
        }
        let mut nodes = vec![(change.outlet.node, InOut::Out(change.outlet.slot))];
        for inlet in model.outlet_successors(change.outlet) {
            nodes.push((inlet.node, InOut::In(inlet.slot)));
        }
        for (node_id, io) in nodes {
            let node = model.node(node_id);
            let more = node.op.change_axes(model, node, io, &change.op)?;
            if more.is_none() {
                return Ok(false);
            }
            let AxisChangeConsequence { substitute_op, wire_changes } = more.unwrap();
            if let Some(op) = substitute_op {
                changed_ops.insert(node.id, op);
            }
            for (wire, op) in wire_changes.into_iter() {
                let outlet = wire.as_outlet(node);
                if !changed_wires.contains_key(&outlet) {
                    changed_wires.insert(outlet, op.clone());
                    todo_changes.push(AxisChange { outlet, op });
                }
            }
        }
    }
    for (node_id, op) in changed_ops.into_iter() {
        model.node_mut(node_id).op = op;
    }
    for (outlet, axis_op) in changed_wires {
        let node = model.node_mut(outlet.node);
        axis_op.apply(&mut node.outputs[outlet.slot].fact.shape)?;
    }
    return Ok(true);
}

#[derive(Debug)]
pub struct ChangeAxes;

impl TypedPass for ChangeAxes {
    fn pass(&self, model: &mut TypedModel) -> TractResult<bool> {
        let mut suggestions = vec![];
        for n in model.eval_order()? {
            let node = model.node(n);
            for suggestion in node.op.suggested_axis_changes()? {
                let outlet = suggestion.0.as_outlet(&node);
                suggestions.push(AxisChange { outlet, op: suggestion.1 })
            }
        }
        for suggestion in suggestions.into_iter() {
            if change_axes(model, &suggestion, true)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

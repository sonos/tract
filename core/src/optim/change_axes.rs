use super::OptimizerSession;
use super::TypedPass;
use crate::internal::*;
use crate::model::*;
use crate::ops::dummy::Dummy;
use crate::ops::einsum::EinSum;
use crate::ops::konst::Const;
use std::collections::hash_map::Entry;
use std::collections::HashSet;
use std::fmt::Debug;

use crate::ops::change_axes::*;

#[derive(Clone, Default)]
pub struct ChangeAxes(HashSet<crate::ops::change_axes::AxisChange>, usize);

impl Debug for ChangeAxes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChangeAxes")
    }
}

impl TypedPass for ChangeAxes {
    fn reset(&mut self) -> TractResult<()> {
        self.0.clear();
        self.1 = 0;
        Ok(())
    }
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut explored: HashSet<AxisChange> = Default::default();
        let mut interfaces = model.output_outlets()?.to_vec();
        interfaces.extend(model.input_outlets()?.iter());
        for node in &model.nodes[self.1..] {
            if node.op_is::<Dummy>() {
                continue;
            }
            for suggestion in node.op.suggested_axis_changes()? {
                let outlet = suggestion.0.as_outlet(node);
                let change = AxisChange { outlet, op: suggestion.1 };
                if self.0.insert(change.clone()) {
                    if let Some((patch, _)) =
                        change_axes(model, &change, &interfaces, &[], &mut explored).with_context(
                            || format!("Making patch for {change:?} from {node}"),
                        )?
                    {
                        self.1 = node.id;
                        return Ok(Some(patch));
                    }
                }
            }
            for (slot, fact) in node.outputs.iter().enumerate() {
                for (ix, dim) in fact.fact.shape.iter().enumerate() {
                    if dim.is_one() {
                        let change =
                            AxisChange { outlet: OutletId::new(node.id, slot), op: AxisOp::Rm(ix) };
                        if self.0.insert(change.clone()) {
                            if let Some((patch, _)) =
                                change_axes(model, &change, &interfaces, &[], &mut explored)
                                    .with_context(|| {
                                        format!("Making patch for {change:?} from {node}")
                                    })?
                            {
                                self.1 = node.id;
                                return Ok(Some(patch));
                            }
                        }
                    }
                }
            }
        }
        Ok(None)
    }
}

#[allow(clippy::type_complexity)]
pub fn change_axes(
    model: &TypedModel,
    change: &AxisChange,
    locked: &[OutletId],
    bounds: &[TVec<OutletId>],
    explored: &mut HashSet<AxisChange>,
) -> TractResult<Option<(TypedModelPatch, TVec<(InOut, AxisOp)>)>> {
    if explored.contains(change) {
        debug!("  Not considering change because deja vu {change:?}");
        return Ok(None);
    }
    if model
        .node(change.outlet.node)
        .op_as::<Const>()
        .is_some_and(|c| c.val().volume() == 1 && c.val().datum_type() != Opaque::datum_type())
    {
        debug!("  Not considering change from const {change:?}");
        return Ok(None);
    }
    debug!("  Considering change {change:?}");
    let mut todo_changes = vec![(change.clone(), None)];
    let mut changed_wires: HashMap<TVec<OutletId>, AxisOp> = HashMap::new();
    let bound_outlets = |o: OutletId| -> TVec<OutletId> {
        bounds.iter().find(|b| b.contains(&o)).cloned().unwrap_or_else(|| tvec!(o))
    };
    changed_wires.insert(bound_outlets(change.outlet), change.op.clone());
    let mut changed_ops: HashMap<usize, Box<dyn TypedOp>> = HashMap::new();
    let mut rewired_scalar_input: HashMap<InletId, (OutletId, AxisOp)> = Default::default();
    while let Some((change, emitter)) = todo_changes.pop() {
        if !explored.insert(change.clone()) {
            return Ok(None);
        }
        let outlet_group = bound_outlets(change.outlet);
        for &outlet in &outlet_group {
            if locked.contains(&outlet) {
                debug!("    Change {change:?} blocked by locked interface {outlet:?}");
                return Ok(None);
            }
            let mut interfaces: Vec<(usize, InOut)> = vec![(outlet.node, InOut::Out(outlet.slot))];
            for inlet in model.outlet_successors(outlet) {
                interfaces.push((inlet.node, InOut::In(inlet.slot)));
            }
            for (node_id, io) in interfaces {
                if Some(node_id) == emitter {
                    continue;
                }
                let node = model.node(node_id);
                // if this is a revisit...
                let op = if let Some(op) = changed_ops.get(&node_id) {
                    trace!("  Change {:?} revisiting {}", change, model.node(node_id));
                    if op.is::<EinSum>() {
                        // FIXME Einsum can swallow any combination of axis change on all interfaces
                        op
                    } else {
                        debug!("  Change {:?} blocked: revisiting {}", change, model.node(node_id));
                        return Ok(None);
                    }
                } else {
                    &node.op
                };
                let more = op
                    .change_axes(model, node, io, &change.op)
                    .with_context(|| format!("Propagating {change:?} to node {node}"))?;
                if more.is_none() {
                    debug!("    Propagation of {change:?} blocked by {node}");
                    return Ok(None);
                }
                let AxisChangeConsequence { substitute_op, wire_changes } = more.unwrap();
                trace!("    Change {:?} enters {} from {:?}", change.op, node, io);
                trace!("       propagates as {wire_changes:?}");
                if let Some(op) = substitute_op {
                    trace!("       replace op by {op:?}");
                    changed_ops.insert(node.id, op);
                }
                for (wire, op) in wire_changes.into_iter() {
                    let outlet = wire.as_outlet(node);
                    // stop upstram propagation to a scalar constant: we will clone it and alter it
                    // at patch generation time
                    if let InOut::In(inlet) = wire {
                        if model
                            .node(outlet.node)
                            .op_as::<Const>()
                            .is_some_and(|k| k.val().volume() == 1)
                        {
                            rewired_scalar_input.insert(InletId::new(node.id, inlet), (outlet, op));
                            continue;
                        }
                    }
                    let outlet_group = bound_outlets(wire.as_outlet(node));
                    match changed_wires.entry(outlet_group.clone()) {
                        Entry::Vacant(entry) => {
                            trace!(
                                "         {wire:?} {op:?} change on {outlet_group:?} is new"
                            );
                            entry.insert(op.clone());
                            todo_changes
                                .push((AxisChange { outlet: outlet_group[0], op }, Some(node_id)));
                        }
                        Entry::Occupied(previous) => {
                            if *previous.get() == op {
                                trace!(
                                    "         {wire:?} {op:?} change on {outlet_group:?} already done"
                                );
                            } else {
                                debug!(
                                    "         {wire:?} {op:?} change on {outlet_group:?} conflicting with {previous:?}. Blocked."
                                    );
                                return Ok(None);
                            }
                        }
                    }
                }
            }
        }
    }
    debug!("Translating {change:?} to patch");
    let mut patch = TypedModelPatch::new(format!("{change:?}"));
    let mut replaced_wires: HashMap<OutletId, OutletId> = HashMap::default();
    let nodes_to_replace = changed_wires
        .keys()
        .flat_map(|outlets| outlets.iter().map(|o| o.node))
        .chain(changed_ops.keys().copied())
        .collect::<std::collections::HashSet<usize>>();
    for node_id in model.eval_order()? {
        let node = model.node(node_id);
        if nodes_to_replace.contains(&node_id) {
            let mut inputs = tvec!();
            for (slot, orig) in node.inputs.iter().enumerate() {
                let tgt = if let Some((outlet, alteration)) =
                    rewired_scalar_input.get(&InletId::new(node_id, slot))
                {
                    let const_node = model.node(outlet.node);
                    let mut value =
                        const_node.op_as::<Const>().unwrap().val().clone().into_tensor();
                    alteration.change_tensor(&mut value, false)?;
                    let name = model.unique_name(&const_node.name);
                    patch.add_const(name, value)?
                } else {
                    *replaced_wires
                        .entry(*orig)
                        .or_insert_with(|| patch.tap_model(model, *orig).unwrap())
                };
                inputs.push(tgt);
            }
            let op: Box<dyn TypedOp> =
                changed_ops.get(&node_id).cloned().unwrap_or_else(|| node.op.clone());
            let new_wires = patch
                .wire_node(&node.name, op.clone(), &inputs)
                .with_context(|| format!("wriring changed_op {op:?}"))?;
            if new_wires.len() == 1
                && patch.node(new_wires[0].node).op_is::<crate::ops::source::TypedSource>()
            {
                patch.inputs.insert(new_wires[0].node, node_id);
            }
            for (ix, w) in new_wires.iter().enumerate() {
                replaced_wires.insert((node_id, ix).into(), *w);
            }
        } else {
            for orig in &node.inputs {
                if let Some(replacement) = replaced_wires.get(orig) {
                    patch.shunt_outside(model, *orig, *replacement)?;
                }
            }
        }
    }
    for output in model.output_outlets()? {
        if let Some(replacement) = replaced_wires.get(output) {
            unsafe {
                patch.shunt_outside_unchecked(*output, *replacement)?;
            }
        }
    }
    let mut interface_change = tvec!();
    for (ix, input) in model.input_outlets()?.iter().enumerate() {
        if let Some(change) = changed_wires.get(&bound_outlets(*input)) {
            interface_change.push((InOut::In(ix), change.clone()));
        }
    }
    for (ix, output) in model.output_outlets()?.iter().enumerate() {
        if let Some(change) = changed_wires.get(&bound_outlets(*output)) {
            interface_change.push((InOut::Out(ix), change.clone()));
        }
    }
    debug!("Patch ready for {change:?}");
    Ok(Some((patch, interface_change)))
}

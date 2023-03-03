use crate::ops::einsum::EinSum;
use crate::ops::konst::Const;
use crate::optim::OptimizerSession;

use super::lir::{LirScan, LirScanOpParams};
use tract_data::internal::*;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub skip: usize,
    pub body: TypedModel,
    decluttered: bool,
    pub seq_length_input_slot: Option<usize>,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
}

impl Scan {
    pub fn to_codegen_op(&self, optimize_inner: bool) -> TractResult<LirScan> {
        let mut model = self.body.clone();
        if optimize_inner {
            model = model.into_optimized()?;
        }
        let plan = SimplePlan::new(model)?;

        Ok(LirScan::new(Arc::new(LirScanOpParams::new(
            self.skip,
            Arc::new(plan),
            self.input_mapping.clone(),
            self.output_mapping.clone(),
        ))))
    }

    pub fn new(
        body: TypedModel,
        input_mapping: Vec<InputMapping>,
        output_mapping: Vec<OutputMapping<TDim>>,
        seq_length_input_slot: Option<usize>,
        skip: usize,
    ) -> TractResult<Scan> {
        body.check_consistency()?;
        ensure!(input_mapping.len() == body.input_outlets()?.len());
        ensure!(output_mapping.len() == body.output_outlets()?.len());
        Ok(Scan {
            skip,
            body,
            decluttered: false,
            input_mapping,
            output_mapping,
            seq_length_input_slot,
        })
    }

    pub fn iteration_count(&self, inputs: &[&TypedFact]) -> Option<TDim> {
        self.to_codegen_op(false).unwrap().iteration_count(inputs)
    }

    fn declutter_body(
        &self,
        session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if !self.decluttered {
            let mut new = self.clone();
            let mut body = self.body.clone();
            session.optimize(&mut body)?;
            new.body = body;
            new.decluttered = true;
            Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, new)?))
        } else {
            Ok(None)
        }
    }

    fn declutter_body_axes(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut suggestions = vec![];
        for n in self.body.eval_order()? {
            let node = self.body.node(n);
            for suggestion in node.op.suggested_axis_changes()? {
                let outlet = suggestion.0.as_outlet(node);
                suggestions.push(AxisChange { outlet, op: suggestion.1 })
            }
        }
        for suggestion in suggestions.into_iter() {
            if let Some(op) =
                self.try_body_axes_change(suggestion, true)?.and_then(|c| c.substitute_op)
            {
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    op,
                )?));
            }
        }
        Ok(None)
    }

    fn remove_outer_input_from_mappings(
        mappings: &[InputMapping],
        discarded: usize,
    ) -> Vec<InputMapping> {
        mappings
            .iter()
            .map(|m| match m {
                &InputMapping::Full { slot } => {
                    InputMapping::Full { slot: slot - (slot > discarded) as usize }
                }
                &InputMapping::Scan(info) => InputMapping::Scan(ScanInfo {
                    slot: info.slot - (info.slot > discarded) as usize,
                    ..info
                }),
                InputMapping::State { initializer } => {
                    let initializer = match initializer {
                        StateInitializer::FromInput(n) => {
                            StateInitializer::FromInput(*n - (*n > discarded) as usize)
                        }
                        StateInitializer::Value(v) => StateInitializer::Value(v.clone()),
                    };
                    InputMapping::State { initializer }
                }
            })
            .collect()
    }

    fn remove_outer_output_from_mappings(
        mappings: &[OutputMapping<TDim>],
        discarded: usize,
    ) -> Vec<OutputMapping<TDim>> {
        mappings
            .iter()
            .map(|m| OutputMapping {
                scan: m.scan.map(|info| ScanInfo {
                    slot: info.slot - (info.slot > discarded) as usize,
                    ..info
                }),
                last_value_slot: m.last_value_slot.map(|n| n - (n > discarded) as usize),
                full_dim_hint: m.full_dim_hint.clone(),
                state: m.state,
            })
            .collect()
    }

    fn declutter_const_input(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        for (body_input_id, mapping) in self.input_mapping.iter().enumerate() {
            if let InputMapping::Full { slot: outer_input_slot } = mapping {
                if let Some(konst) = inputs[*outer_input_slot].konst.as_ref() {
                    let mut op = self.clone();
                    let src = op.body.inputs[body_input_id];
                    op.body.inputs.remove(body_input_id);
                    op.body.nodes[src.node].inputs.clear();
                    op.body.nodes[src.node].op = Box::new(Const::new(konst.clone()));
                    op.input_mapping.remove(body_input_id);
                    op.input_mapping = Self::remove_outer_input_from_mappings(
                        &op.input_mapping,
                        *outer_input_slot,
                    );
                    let mut inputs = node.inputs.clone();
                    inputs.remove(*outer_input_slot);
                    return Ok(Some(TypedModelPatch::replace_single_op(model, node, &inputs, op)?));
                }
            }
        }
        Ok(None)
    }

    fn declutter_const_initializer(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        for (ix, mapping) in self.input_mapping.iter().enumerate() {
            if let InputMapping::State { initializer: StateInitializer::FromInput(n) } = mapping {
                if let Some(i) = inputs[*n].konst.as_ref() {
                    let mut op = self.clone();
                    op.input_mapping[ix] =
                        InputMapping::State { initializer: StateInitializer::Value(i.clone()) };
                    op.input_mapping =
                        Self::remove_outer_input_from_mappings(&op.input_mapping, *n);
                    let mut inputs = node.inputs.clone();
                    inputs.remove(*n);
                    return Ok(Some(TypedModelPatch::replace_single_op(model, node, &inputs, op)?));
                }
            }
        }
        Ok(None)
    }

    fn declutter_discard_unused_input_mapping(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (inner_input_id, input) in self.body.input_outlets()?.iter().enumerate() {
            let source_node = self.body.node(input.node);
            if source_node.outputs[0].successors.len() == 0
                && !self.body.output_outlets()?.contains(input)
            {
                let mut new_inputs = node.inputs.clone();
                let slot = match &self.input_mapping[inner_input_id] {
                    InputMapping::Full { slot } => Some(*slot),
                    InputMapping::Scan(info) => Some(info.slot),
                    InputMapping::State { initializer } => match initializer {
                        StateInitializer::FromInput(n) => Some(*n),
                        _ => None,
                    },
                };
                let mut new_mappings: Vec<_> = self.input_mapping.clone();
                new_mappings.remove(inner_input_id);
                if let Some(slot) = slot {
                    new_mappings = Self::remove_outer_input_from_mappings(&new_mappings, slot);
                }
                let mut model_inputs = self.body.input_outlets()?.to_vec();
                if let Some(slot) = slot {
                    new_inputs.remove(slot);
                }
                model_inputs.remove(inner_input_id);
                let mut body = self.body.clone();
                let mut patch = TypedModelPatch::default();
                patch.obliterate(source_node.id)?;
                patch.apply(&mut body)?;
                body.set_input_outlets(&model_inputs)?;
                body.declutter()?;
                let op = Self {
                    body,
                    skip: self.skip,
                    seq_length_input_slot: self.seq_length_input_slot,
                    input_mapping: new_mappings,
                    decluttered: true,
                    output_mapping: self.output_mapping.clone(),
                };
                return Ok(Some(TypedModelPatch::replace_single_op(model, node, &new_inputs, op)?));
            }
        }
        Ok(None)
    }

    fn declutter_discard_useless_outer_output(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (ix, o) in node.outputs.iter().enumerate() {
            if o.successors.len() == 0
                && !model.output_outlets()?.contains(&OutletId::new(node.id, ix))
            {
                let mappings = self
                    .output_mapping
                    .iter()
                    .map(|m| OutputMapping {
                        scan: m.scan.filter(|info| info.slot != ix),
                        last_value_slot: m.last_value_slot.filter(|s| *s != ix),
                        full_dim_hint: m.full_dim_hint.clone(),
                        state: m.state,
                    })
                    .collect::<Vec<_>>();
                let mut op = self.clone();
                op.output_mapping = Self::remove_outer_output_from_mappings(&mappings, ix);
                let mut patch = TypedModelPatch::default();
                let inputs = node
                    .inputs
                    .iter()
                    .map(|&i| patch.tap_model(model, i))
                    .collect::<TractResult<Vec<_>>>()?;
                let wires = patch.wire_node(&*node.name, op, &inputs)?;
                for oix in 0..node.outputs.len() {
                    if oix != ix {
                        patch.shunt_outside(
                            model,
                            OutletId::new(node.id, oix),
                            wires[oix - (oix > ix) as usize],
                        )?;
                    }
                }
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    fn declutter_discard_empty_output_mapping_with_body_output(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (ix, om) in self.output_mapping.iter().enumerate() {
            if om.last_value_slot.is_none() && om.scan.is_none() && !om.state {
                let mut new_op = self.clone();
                new_op.output_mapping.remove(ix);
                new_op.body.outputs.remove(ix);
                new_op.decluttered = false;
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    new_op,
                )?));
            }
        }
        Ok(None)
    }

    fn declutter_pull_batcheable_input(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        'candidate: for (model_input_ix, input) in self.input_mapping.iter().enumerate() {
            if let Some(scan_info) = input.as_scan() {
                let scan_source = self.body.input_outlets()?[model_input_ix];
                let scan_source_node = self.body.node(scan_source.node);
                for mut succ in &scan_source_node.outputs[0].successors {
                    for &succ_input in &self.body.node(succ.node).inputs {
                        if succ_input != scan_source
                            && self.body.outlet_fact(succ_input)?.konst.is_none()
                        {
                            continue 'candidate;
                        }
                    }
                    if self.body.node(succ.node).outputs.len() != 1 {
                        continue;
                    }
                    let mut new_body = self.body.clone();
                    // insert propagate axis on einsum
                    if let Some(einsum) = new_body.node(succ.node).op_as::<EinSum>() {
                        if let Some(patch) = einsum
                            .propagate_axis(
                                &new_body,
                                new_body.node(succ.node),
                                InOut::In(succ.slot),
                                scan_info.axis,
                            )
                            .context("building axis propagating patch")?
                        {
                            patch.apply(&mut new_body)?;
                            // propagate axis injects new nodes at the end. last successor of input
                            // in new net will be the new succ
                            let new_body_scan_input = new_body.input_outlets()?[model_input_ix];
                            succ = &new_body.node(new_body_scan_input.node).outputs[0]
                                .successors
                                .last()
                                .unwrap();
                        }
                    }

                    let axes_mapping = {
                        let (input_facts, output_facts) =
                            new_body.node_facts(new_body.node(succ.node).id)?;
                        new_body.node(succ.node).op.axes_mapping(&input_facts, &output_facts)?
                    };
                    let axis_info = axes_mapping.input_axis(succ.slot, scan_info.axis)?;
                    if let &[axis_after] = &*axis_info.outputs[0] {
                        let mut outside_patch = TypedModelPatch::new(format!(
                            "Outer patch for input extraction of {}",
                            new_body.node(succ.node)
                        ));
                        let mut patch_inputs = node
                            .inputs
                            .iter()
                            .map(|&i| outside_patch.tap_model(model, i))
                            .collect::<TractResult<TVec<_>>>()?;
                        let mut extracted_op_inputs = tvec!();
                        for (ix, outlet) in new_body.node(succ.node).inputs.iter().enumerate() {
                            let wire = if ix == succ.slot {
                                patch_inputs[scan_info.slot]
                            } else if let Some(konst) =
                                new_body.outlet_fact(*outlet)?.konst.as_ref()
                            {
                                outside_patch.add_const(
                                    format!(
                                        "{}.extracted.{}",
                                        node.name,
                                        new_body.node(outlet.node).name
                                    ),
                                    konst.clone(),
                                )?
                            } else {
                                unreachable!();
                            };
                            extracted_op_inputs.push(wire);
                        }
                        let new_input_wire = outside_patch.wire_node(
                            format!("{}.extracted.{}", node.name, new_body.node(succ.node).name),
                            new_body.node(succ.node).op.clone(),
                            &extracted_op_inputs,
                        )?[0];
                        patch_inputs.push(new_input_wire);
                        let new_input_outer_fact = outside_patch.outlet_fact(new_input_wire)?;
                        let mut new_input_inner_fact = new_input_outer_fact.clone();
                        new_input_inner_fact.shape.set(axis_after, scan_info.chunk.abs().to_dim());

                        let mut new_body = new_body.clone();
                        let new_source_wire = new_body.add_source(
                            format!("{}.extracted.{}", node.name, new_body.node(succ.node).name),
                            new_input_inner_fact,
                        )?;
                        let mut inner_patch = TypedModelPatch::new(format!(
                            "Inner body patch for extraction of {}",
                            new_body.node(succ.node)
                        ));
                        let new_source_wire_in_patch =
                            inner_patch.tap_model(&new_body, new_source_wire)?;
                        inner_patch
                            .shunt_outside(
                                &new_body,
                                OutletId::new(succ.node, 0),
                                new_source_wire_in_patch,
                            )
                            .with_context(|| "patching inner model")?;
                        inner_patch.apply(&mut new_body)?;

                        let mut input_mapping = self.input_mapping.clone();
                        input_mapping.push(InputMapping::Scan(ScanInfo {
                            axis: axis_after,
                            chunk: scan_info.chunk,
                            slot: node.inputs.len(),
                        }));

                        let new_op = Self {
                            input_mapping,
                            output_mapping: self.output_mapping.clone(),
                            decluttered: false,
                            body: new_body,
                            skip: self.skip,
                            seq_length_input_slot: self.seq_length_input_slot,
                        };
                        let output_wires =
                            outside_patch.wire_node(&*node.name, new_op, &patch_inputs)?;
                        for w in output_wires {
                            outside_patch
                                .shunt_outside(model, OutletId::new(node.id, w.slot), w)
                                .with_context(|| "patching outer model")?;
                        }
                        return Ok(Some(outside_patch));
                    }
                }
            }
        }
        Ok(None)
    }

    fn declutter_pull_constant_outputs(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (model_output_ix, mapping) in self.output_mapping.iter().enumerate() {
            if let Some(slot) = mapping.last_value_slot {
                if let Some(k) = self.body.output_fact(model_output_ix)?.konst.clone() {
                    let inner_node = self.body.output_outlets()?[model_output_ix].node;
                    let inner_node = self.body.node(inner_node);
                    let mut patch =
                        TypedModelPatch::new(format!("Extract const node {inner_node}"));
                    let cst = patch.add_const(format!("{}.{}", &node.name, &inner_node.name), k)?;
                    patch.shunt_outside(model, OutletId::new(node.id, slot), cst)?;
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }

    fn declutter_pull_batcheable_output(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (model_ix, mapping) in self.output_mapping.iter().enumerate() {
            if let Some(scan_info) = mapping.scan {
                let emitter_outlet = self.body.output_outlets()?[model_ix];
                if self.body.node(emitter_outlet.node).outputs[emitter_outlet.slot].successors.len()
                    > 0
                    || self.body.inputs.contains(&emitter_outlet)
                    || mapping.state
                    || mapping.scan.map(|i| i.chunk > 1).unwrap_or(true)
                {
                    // continue if both last_value and full values are exported
                    continue;
                }
                let mut new_body = self.body.clone();
                if let Some(einsum) = new_body.node(emitter_outlet.node).op_as::<EinSum>() {
                    if let Some(patch) = einsum
                        .propagate_axis(
                            &new_body,
                            new_body.node(emitter_outlet.node),
                            InOut::Out(0),
                            scan_info.axis,
                        )
                        .context("building axis propagating patch")?
                    {
                        patch.apply(&mut new_body)?;
                    }
                }
                let emitter_outlet = new_body.output_outlets()?[model_ix];
                let invariants = {
                    let (input_facts, output_facts) = new_body.node_facts(emitter_outlet.node)?;
                    new_body
                        .node(emitter_outlet.node)
                        .op
                        .axes_mapping(&input_facts, &output_facts)?
                };
                let axis_tracking = invariants.output_axis(emitter_outlet.slot, scan_info.axis)?;
                if axis_tracking.outputs.iter().any(|o| o.len() > 1) {
                    return Ok(None);
                }
                let mut new_output_mapping = self.output_mapping.clone();
                let mut new_scan_outputs = node.outputs.len();
                let mut outer_slots = vec![];

                for (input_slot, input) in
                    new_body.node(emitter_outlet.node).inputs.clone().iter().enumerate()
                {
                    if new_body.outputs.iter().all(|o| o != input) {
                        new_output_mapping.push(OutputMapping::default());
                        new_body.outputs.push(*input);
                    }
                    let body_output_id = new_body.outputs.iter().position(|o| o == input).unwrap();
                    let mut mapping = &mut new_output_mapping[body_output_id];
                    let outer_slot = if new_body.outlet_fact(*input)?.konst.is_some() {
                        if mapping.last_value_slot.is_none() {
                            mapping.last_value_slot = Some(new_scan_outputs);
                        }
                        new_scan_outputs += 1;
                        mapping.last_value_slot.unwrap()
                    } else if let &[axis] = &*axis_tracking.inputs[input_slot] {
                        if mapping.scan.is_none() {
                            mapping.scan = Some(ScanInfo {
                                slot: new_scan_outputs,
                                axis,
                                chunk: scan_info.chunk,
                            });
                            new_scan_outputs += 1;
                        }
                        mapping.scan.unwrap().slot
                    } else {
                        return Ok(None);
                    };
                    outer_slots.push(outer_slot);
                }
                let mut outside_patch = TypedModelPatch::new(format!(
                    "Outside patch for output extraction of {}",
                    new_body.node(emitter_outlet.node)
                ));
                let inputs = node
                    .inputs
                    .iter()
                    .map(|&i| outside_patch.tap_model(model, i))
                    .collect::<TractResult<TVec<_>>>()?;
                let new_op = Self {
                    input_mapping: self.input_mapping.clone(),
                    output_mapping: new_output_mapping,
                    decluttered: false,
                    body: new_body.clone(), // FIXME maybe remove clone
                    skip: self.skip,
                    seq_length_input_slot: self.seq_length_input_slot,
                };
                let scan_outputs = outside_patch.wire_node(&node.name, new_op, &inputs)?;
                let output = mapping.scan.unwrap();
                let inputs =
                    outer_slots.iter().map(|slot| scan_outputs[*slot]).collect::<TVec<_>>();
                let wire = outside_patch.wire_node(
                    &new_body.node(emitter_outlet.node).name,
                    new_body.node(emitter_outlet.node).op.clone(),
                    &inputs,
                )?[0];
                outside_patch.shunt_outside(model, OutletId::new(node.id, output.slot), wire)?;
                for output_slot in 0..node.outputs.len() {
                    if output_slot != output.slot {
                        outside_patch.shunt_outside(
                            model,
                            OutletId::new(node.id, output_slot),
                            OutletId::new(scan_outputs[0].node, output_slot),
                        )?;
                    }
                }
                return Ok(Some(outside_patch));
            }
        }
        Ok(None)
    }

    fn body_bounds(&self) -> TractResult<TVec<TVec<OutletId>>> {
        let input_state_outlets = self
            .input_mapping
            .iter()
            .zip(self.body.input_outlets()?.iter())
            .filter(|(m, _)| m.as_state().is_some())
            .map(|(_, o)| o);
        let output_state_outlets = self
            .output_mapping
            .iter()
            .zip(self.body.output_outlets()?.iter())
            .filter(|(m, _)| m.state)
            .map(|(_, o)| o);
        Ok(input_state_outlets.zip(output_state_outlets).map(|(&i, &o)| tvec!(i, o)).collect())
    }

    fn body_exposed_outlets(&self) -> TractResult<TVec<OutletId>> {
        let input_outlets = self
            .input_mapping
            .iter()
            .zip(self.body.input_outlets()?.iter())
            .filter(|(m, _)| !m.invisible())
            .map(|(_, o)| o);
        let output_outlets = self
            .output_mapping
            .iter()
            .zip(self.body.output_outlets()?.iter())
            .filter(|(m, _)| !m.invisible())
            .map(|(_, o)| o);
        Ok(input_outlets.chain(output_outlets).cloned().collect())
    }

    fn try_body_axes_change(
        &self,
        change: AxisChange,
        locked_interface: bool,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        self.body.check_consistency()?;
        let interface = self.body_exposed_outlets()?;
        let (patch, body_changed_wires) = if let Some(changes) =
            crate::ops::change_axes::change_axes(
                &self.body,
                &change,
                if locked_interface { &interface } else { &[] },
                &self.body_bounds()?,
            )? {
            changes
        } else {
            return Ok(None);
        };
        let mut body = self.body.clone();
        patch.apply(&mut body)?;
        body.compact()?;
        let mut wire_changes = tvec!();
        let mut input_mapping: Vec<InputMapping> = self.input_mapping.clone();
        for (ix, m) in input_mapping.iter_mut().enumerate() {
            if let Some(change) = body_changed_wires
                .iter()
                .find(|(iface, _change)| iface == &InOut::In(ix))
                .map(|pair| pair.1.clone())
            {
                if let Some(slot) = m.outer_slot() {
                    wire_changes.push((InOut::In(slot), change.clone()));
                }
                match &*m {
                    InputMapping::Full { .. } => (),
                    &InputMapping::Scan(info) => {
                        if let Some(axis) = change.transform_axis(info.axis) {
                            *m = InputMapping::Scan(ScanInfo { axis, ..info });
                        } else {
                            return Ok(None);
                        };
                    }
                    InputMapping::State { initializer } => match initializer {
                        StateInitializer::FromInput(_) => (),
                        StateInitializer::Value(ref v) => {
                            let mut v = v.clone().into_tensor();
                            change.change_tensor(&mut v, false)?;
                            *m = InputMapping::State {
                                initializer: StateInitializer::Value(v.into_arc_tensor()),
                            };
                        }
                    },
                };
            }
        }
        let mut output_mapping: Vec<OutputMapping<TDim>> = self.output_mapping.clone();
        for (ix, m) in output_mapping.iter_mut().enumerate() {
            if let Some(change) = body_changed_wires
                .iter()
                .find(|(iface, _change)| iface == &InOut::Out(ix))
                .map(|pair| pair.1.clone())
            {
                if let Some(info) = m.scan.as_mut() {
                    if let Some(new_axis) = change.transform_axis(info.axis) {
                        info.axis = new_axis;
                    } else {
                        return Ok(None);
                    }
                    wire_changes.push((InOut::Out(info.slot), change.clone()));
                }
                if let Some(slot) = m.last_value_slot {
                    wire_changes.push((InOut::Out(slot), change.clone()));
                }
            };
        }
        body.check_consistency()?;
        let op = Some(Box::new(Scan {
            body,
            input_mapping,
            output_mapping,
            decluttered: false,
            ..self.clone()
        }) as _);
        Ok(Some(AxisChangeConsequence { substitute_op: op, wire_changes }))
    }
}

impl Op for Scan {
    fn name(&self) -> Cow<str> {
        "Scan".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut lines = vec![];
        for (ix, im) in self.input_mapping.iter().enumerate() {
            lines.push(format!("Model input  #{ix}: {im:?}"));
        }
        for (ix, om) in self.output_mapping.iter().enumerate() {
            lines.push(format!("Model output #{ix}: {om:?}"));
        }
        Ok(lines)
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for Scan {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        self.to_codegen_op(false)?.state(session, node_id)
    }
}

impl TypedOp for Scan {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut outputs = tvec!();
        let iters = {
            let info = self.input_mapping.iter().flat_map(|it| it.as_scan()).next().unwrap();
            inputs[info.slot].shape[info.axis].clone().div_ceil(info.chunk.unsigned_abs() as u64)
        };
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.body.output_fact(ix)?;
            if let Some(info) = output.scan {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[info.axis].clone() * &iters);
                shape.set(info.axis, scanning_dim);
                outputs.push((info.slot, fact.datum_type.fact(shape)));
            }
            if let Some(slot) = output.last_value_slot {
                outputs.push((slot, fact.datum_type.fact(fact.shape.clone())));
            }
        }
        outputs.sort_by_key(|a| a.0);
        anyhow::ensure!(outputs.iter().enumerate().all(|(ix, (slot, _))| ix == *slot));
        let outputs: TVec<_> = outputs.into_iter().map(|(_slot, v)| v).collect();
        Ok(outputs)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut mappings = vec![];
        let body_invs = self.body.axes_mapping().with_context(|| "Computing body axes mapping")?;
        for body_axis in body_invs.iter_all_axes() {
            let mut info = Axis::new(body_axis.repr, inputs.len(), outputs.len());
            for (ix, input_mapping) in self.input_mapping.iter().enumerate() {
                if let Some(slot) = input_mapping.outer_slot() {
                    info.inputs[slot] = body_axis.inputs[ix].clone();
                }
            }
            for (ix, output_mapping) in self.output_mapping.iter().enumerate() {
                let mut slots = vec![];
                if let Some(scan) = output_mapping.scan {
                    slots.push(scan.slot);
                }
                if let Some(slot) = output_mapping.last_value_slot {
                    slots.push(slot);
                }
                for slot in slots {
                    info.outputs[slot] = body_axis.outputs[ix].clone();
                }
            }
            if info.inputs.iter().any(|i| i.len() > 0) || info.outputs.iter().any(|i| i.len() > 0) {
                mappings.push(info);
            }
        }
        mappings.into_iter().collect()
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        let mut suggestions = tvec!();
        for input in &self.input_mapping {
            if let InputMapping::Scan(info) = input {
                if info.axis != 0 {
                    suggestions.push((InOut::In(info.slot), AxisOp::Move(info.axis, 0)))
                }
            }
        }
        for output in &self.output_mapping {
            if let Some(scan) = output.scan {
                if scan.axis != 0 {
                    suggestions.push((InOut::Out(scan.slot), AxisOp::Move(scan.axis, 0)))
                }
            }
        }
        Ok(suggestions)
    }

    fn change_axes(
        &self,
        _model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        trace!("Propagating through {}: {:?} {:?}", node, io, change);
        let body_leading_outlet = match io {
            InOut::In(ix) => {
                if let Some(input) =
                    self.input_mapping.iter().position(|im| im.outer_slot() == Some(ix))
                {
                    self.body.input_outlets()?[input]
                } else {
                    trace!("Node {} blocking (input mapping) ", node);
                    return Ok(None);
                }
            }
            InOut::Out(slot) => {
                let output = self
                    .output_mapping
                    .iter()
                    .position(|im| {
                        im.scan.map(|i| i.slot) == Some(slot) || im.last_value_slot == Some(slot)
                    })
                    .unwrap();
                self.body.output_outlets()?[output]
            }
        };
        let axis_change = AxisChange { outlet: body_leading_outlet, op: change.clone() };
        let result = self
            .try_body_axes_change(axis_change, false)
            .with_context(|| format!("Attemping to run change through scan body"))?;
        if result.is_some() {
            trace!("{} accepted axis change", node);
        } else {
            trace!("{} rejected axis change", node);
        }
        Ok(result)
    }

    fn declutter_with_session(
        &self,
        session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        macro_rules! pass {
            ($func:ident) => {
                if let Some(mut r) = self
                    .$func(session, model, node)
                    .with_context(|| format!("{}", stringify!($func)))?
                {
                    trace!(stringify!($func));
                    r.push_context(stringify!($func));
                    return Ok(Some(r));
                }
            };
        }
        pass!(declutter_const_initializer);
        pass!(declutter_const_input);
        pass!(declutter_discard_unused_input_mapping);
        pass!(declutter_discard_useless_outer_output);
        pass!(declutter_discard_empty_output_mapping_with_body_output);
        pass!(declutter_body);
        pass!(declutter_body_axes);
        pass!(declutter_pull_constant_outputs);
        pass!(declutter_pull_batcheable_input);
        pass!(declutter_pull_batcheable_output);
        Ok(None)
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|o| mapping[o]).collect::<TVec<_>>();
        let op = Self {
            output_mapping: self
                .output_mapping
                .iter()
                .map(|om| om.concretize_dims(values))
                .collect::<TractResult<Vec<_>>>()?,
            body: self.body.concretize_dims(values)?,
            ..self.clone()
        };
        target.wire_node(&node.name, op, &inputs)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            self.to_codegen_op(true)?,
        )?))
    }
}

use std::collections::HashSet;

use crate::ops::einsum::EinSum;
use crate::ops::konst::Const;
use crate::optim::OptimizerSession;

use super::optimized::{OptScan, ScanOpParams};
use tract_data::internal::*;

use super::*;

#[derive(Debug, Clone, Default)]
pub struct Scan {
    pub skip: usize,
    pub reset_every_turn: bool,
    pub body: TypedModel,
    pub decluttered: bool,
    pub input_mapping: Vec<InputMapping>,
    pub output_mapping: Vec<OutputMapping<TDim>>,
}

impl Scan {
    pub fn to_codegen_op(&self, optimize_inner: bool) -> TractResult<OptScan> {
        let mut model = self.body.clone();
        if optimize_inner {
            model = model.into_optimized()?;
        }
        let plan = SimplePlan::new(model)?;

        Ok(OptScan::new(Arc::new(ScanOpParams::new(
            self.skip,
            self.reset_every_turn,
            Arc::new(plan),
            self.input_mapping.clone(),
            self.output_mapping.clone(),
        ))))
    }

    pub fn new(
        body: TypedModel,
        input_mapping: Vec<InputMapping>,
        output_mapping: Vec<OutputMapping<TDim>>,
        skip: usize,
    ) -> TractResult<Scan> {
        body.check_consistency()?;
        ensure!(input_mapping.len() == body.input_outlets()?.len());
        ensure!(output_mapping.len() == body.output_outlets()?.len());
        Ok(Scan {
            skip,
            reset_every_turn: false,
            body,
            decluttered: false,
            input_mapping,
            output_mapping,
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
            for (slot, fact) in node.outputs.iter().enumerate() {
                for (ix, dim) in fact.fact.shape.iter().enumerate() {
                    if dim.is_one() {
                        suggestions.push(AxisChange {
                            outlet: OutletId::new(n, slot),
                            op: AxisOp::Rm(ix),
                        });
                    }
                }
            }
        }
        let node_input_facts = model.node_input_facts(node.id)?;
        for suggestion in suggestions.into_iter() {
            if let Some(conseq) = self.try_body_axes_change(suggestion, true, &node_input_facts)? {
                let mut patch = TypedModelPatch::default();
                let mut inputs = tvec!();
                for outlet in &node.inputs {
                    inputs.push(patch.tap_model(model, *outlet)?);
                }
                for change in conseq.wire_changes {
                    if let InOut::In(i) = change.0 {
                        let mut value = patch
                            .outlet_fact(inputs[i])?
                            .konst
                            .clone()
                            .context("Will only reshape constants")?
                            .into_tensor();
                        change.1.change_tensor(&mut value, false)?;
                        let konst_name = patch.node(inputs[i].node).name.clone();
                        inputs[i] = patch.add_const(konst_name, value)?;
                    }
                }
                let wires = patch.wire_node(
                    &node.name,
                    conseq.substitute_op.unwrap_or_else(|| Box::new(self.clone())),
                    &inputs,
                )?;
                for (ix, new) in wires.into_iter().enumerate() {
                    patch.shunt_outside(model, OutletId::new(node.id, ix), new)?;
                }
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }

    fn remove_outer_output_from_mappings(
        mappings: &[OutputMapping<TDim>],
        discarded: usize,
    ) -> Vec<OutputMapping<TDim>> {
        mappings
            .iter()
            .map(|m| OutputMapping {
                scan: m.scan.map(|(slot, info)| (slot - (slot > discarded) as usize, info)),
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
        for (slot, mapping) in self.input_mapping.iter().enumerate() {
            if let InputMapping::Full = mapping {
                if let Some(konst) = inputs[slot].konst.as_ref() {
                    let mut op = self.clone();
                    let src = op.body.inputs[slot];
                    op.body.inputs.remove(slot);
                    op.body.nodes[src.node].inputs.clear();
                    op.body.nodes[src.node].op = Box::new(Const::new(konst.clone())?);
                    op.input_mapping.remove(slot);
                    let mut inputs = node.inputs.clone();
                    inputs.remove(slot);
                    return Ok(Some(TypedModelPatch::replace_single_op(model, node, &inputs, op)?));
                }
            }
        }
        Ok(None)
    }

    fn declutter_discard_unused_input(
        &self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        for (slot, input) in self.body.input_outlets()?.iter().enumerate() {
            let source_node = self.body.node(input.node);
            if source_node.outputs[0].successors.len() == 0
                && !self.body.output_outlets()?.contains(input)
            {
                let mut new_inputs = node.inputs.clone();
                new_inputs.remove(slot);
                let mut new_mappings: Vec<_> = self.input_mapping.clone();
                new_mappings.remove(slot);
                let mut model_inputs = self.body.input_outlets()?.to_vec();
                model_inputs.remove(slot);
                let mut body = self.body.clone();
                let mut patch = TypedModelPatch::default();
                patch.obliterate(source_node.id)?;
                patch.apply(&mut body)?;
                body.set_input_outlets(&model_inputs)?;
                body.declutter()?;
                let op =
                    Self { body, input_mapping: new_mappings, decluttered: true, ..self.clone() };
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
                        scan: m.scan.filter(|(slot, _info)| *slot != ix),
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
        'candidate: for (slot, input) in self.input_mapping.iter().enumerate() {
            if let Some(scan_info) = input.as_scan() {
                let scan_source = self.body.input_outlets()?[slot];
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
                            new_body.compute_const_facts()?;
                            // propagate axis injects new nodes at the end. last successor of input
                            // in new net will be the new succ
                            let new_body_scan_input = new_body.input_outlets()?[slot];
                            succ = new_body.node(new_body_scan_input.node).outputs[0]
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
                    let axis_info = axes_mapping.axis((InOut::In(succ.slot), scan_info.axis))?;
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
                                patch_inputs[slot]
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
                        }));

                        let new_op = Self {
                            input_mapping,
                            decluttered: false,
                            body: new_body,
                            ..self.clone()
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
        for (mapping_ix, mapping) in self.output_mapping.iter().enumerate() {
            if let Some((_, scan_info)) = mapping.scan {
                let emitter_outlet = self.body.output_outlets()?[mapping_ix];
                if self.body.node(emitter_outlet.node).outputs[emitter_outlet.slot].successors.len()
                    > 0
                    || self.body.inputs.contains(&emitter_outlet)
                    || mapping.state
                    || mapping.scan.map(|(_slot, i)| i.chunk > 1).unwrap_or(true)
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
                        new_body.prop_consts()?;
                    }
                }
                let emitter_outlet = new_body.output_outlets()?[mapping_ix];
                let invariants = {
                    let (input_facts, output_facts) = new_body.node_facts(emitter_outlet.node)?;
                    new_body
                        .node(emitter_outlet.node)
                        .op
                        .axes_mapping(&input_facts, &output_facts)?
                };
                let axis_tracking =
                    invariants.axis((InOut::Out(emitter_outlet.slot), scan_info.axis))?;
                if axis_tracking.outputs.iter().any(|o| o.len() > 1) {
                    return Ok(None);
                }
                let mut new_output_mapping = self.output_mapping.clone();
                let mut new_scan_outputs = node.outputs.len();
                let mut outer_slots = vec![];

                // rewire input of the extracted node through the scan outlet boundary
                for (input_slot, input) in
                    new_body.node(emitter_outlet.node).inputs.clone().iter().enumerate()
                {
                    if new_body.outputs.iter().all(|o| o != input) {
                        new_output_mapping.push(OutputMapping::default());
                        new_body.outputs.push(*input);
                    }
                    let body_output_id = new_body.outputs.iter().position(|o| o == input).unwrap();
                    let mapping = &mut new_output_mapping[body_output_id];
                    let outer_slot = if new_body.outlet_fact(*input)?.konst.is_some() {
                        if mapping.last_value_slot.is_none() {
                            mapping.last_value_slot = Some(new_scan_outputs);
                            new_scan_outputs += 1;
                        }
                        mapping.last_value_slot.unwrap()
                    } else if let &[axis] = &*axis_tracking.inputs[input_slot] {
                        if mapping.scan.is_none() {
                            mapping.scan =
                                Some((new_scan_outputs, ScanInfo { axis, chunk: scan_info.chunk }));
                            new_scan_outputs += 1;
                        }
                        mapping.scan.unwrap().0
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
                    output_mapping: new_output_mapping,
                    decluttered: false,
                    body: new_body.clone(), // FIXME maybe remove clone
                    ..self.clone()
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
                outside_patch.shunt_outside(model, OutletId::new(node.id, output.0), wire)?;
                for output_slot in 0..node.outputs.len() {
                    if output_slot != output.0 {
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
            .filter(|(m, _)| m.is_state())
            .map(|(_, o)| o);
        let output_state_outlets = self
            .output_mapping
            .iter()
            .zip(self.body.output_outlets()?.iter())
            .filter(|(m, _)| m.state)
            .map(|(_, o)| o);
        Ok(input_state_outlets.zip(output_state_outlets).map(|(&i, &o)| tvec!(i, o)).collect())
    }

    fn body_locked_outlets(&self, node_input_facts: &[&TypedFact]) -> TractResult<TVec<OutletId>> {
        let input_outlets =
            self.body.input_outlets()?.iter().enumerate().filter_map(|(slot, o)| {
                if node_input_facts[slot].konst.is_none() {
                    Some(o)
                } else {
                    None
                }
            });
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
        node_input_facts: &[&TypedFact],
    ) -> TractResult<Option<AxisChangeConsequence>> {
        self.body.check_consistency()?;
        let locked_outlets = self.body_locked_outlets(node_input_facts)?;
        let mut explored: HashSet<AxisChange> = Default::default();
        let (body_patch, body_changed_wires) = if let Some(changes) =
            crate::optim::change_axes::change_axes(
                &self.body,
                &change,
                if locked_interface { &locked_outlets } else { &[] },
                &self.body_bounds()?,
                &mut explored,
            )? {
            changes
        } else {
            return Ok(None);
        };
        let mut body = self.body.clone();
        body_patch.apply(&mut body)?;
        body.compact()?;
        let mut wire_changes = tvec!();
        let mut input_mapping: Vec<InputMapping> = self.input_mapping.clone();
        for (slot, m) in input_mapping.iter_mut().enumerate() {
            if let Some(change) = body_changed_wires
                .iter()
                .find(|(iface, _change)| iface == &InOut::In(slot))
                .map(|pair| pair.1.clone())
            {
                wire_changes.push((InOut::In(slot), change.clone()));
                if let InputMapping::Scan(info) = m {
                    if let Some(axis) = change.transform_axis(info.axis) {
                        info.axis = axis;
                    } else {
                        return Ok(None);
                    };
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
                if let Some((slot, info)) = m.scan.as_mut() {
                    if let Some(new_axis) = change.transform_axis(info.axis) {
                        info.axis = new_axis;
                    } else {
                        return Ok(None);
                    }
                    wire_changes.push((InOut::Out(*slot), change.clone()));
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
        lines.push(format!("skip:{} reset_every_turn:{:?}", self.skip, self.reset_every_turn));
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
        anyhow::ensure!(inputs.len() == self.body.inputs.len());
        anyhow::ensure!(self.input_mapping.len() == self.body.inputs.len());
        anyhow::ensure!(
            self.input_mapping.iter().filter(|m| m.is_state()).count()
                == self.output_mapping.iter().filter(|m| m.state).count()
        );
        for (i, o) in
            self.input_mapping.iter().enumerate().filter(|(_, m)| m.is_state()).map(|(i, _)| i).zip(
                self.output_mapping.iter().enumerate().filter(|(_, m)| m.state).map(|(o, _)| o),
            )
        {
            let ifact = self.body.outlet_fact(self.body.inputs[i])?;
            let ofact = self.body.outlet_fact(self.body.outputs[o])?;
            anyhow::ensure!(ifact == ofact,
                "inconsistent state shape: body input {i} is {ifact:?} and body output {o} is {ofact:?}",
            )
        }
        let mut outputs = tvec!();
        let iters = super::iteration_count(&self.input_mapping, inputs).context("No scan input")?;
        for (ix, output) in self.output_mapping.iter().enumerate() {
            let fact = self.body.output_fact(ix)?;
            if let Some((slot, info)) = output.scan {
                let mut shape = fact.shape.clone();
                let scanning_dim =
                    output.full_dim_hint.clone().unwrap_or(shape[info.axis].clone() * &iters);
                shape.set(info.axis, scanning_dim);
                outputs.push((slot, fact.datum_type.fact(shape)));
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
            info.inputs.clone_from(&body_axis.inputs);
            for (ix, output_mapping) in self.output_mapping.iter().enumerate() {
                let mut slots = vec![];
                if let Some((slot, _scan)) = output_mapping.scan {
                    slots.push(slot);
                }
                if let Some(slot) = output_mapping.last_value_slot {
                    slots.push(slot);
                }
                for slot in slots {
                    info.outputs[slot].clone_from(&body_axis.outputs[ix]);
                }
            }
            if info.inputs.iter().any(|i| i.len() > 0) || info.outputs.iter().any(|i| i.len() > 0) {
                mappings.push(info);
            }
        }
        AxesMapping::new(inputs.len(), outputs.len(), mappings)
    }

    fn suggested_axis_changes(&self) -> TractResult<TVec<(InOut, AxisOp)>> {
        let mut suggestions = tvec!();
        for (slot, input) in self.input_mapping.iter().enumerate() {
            if let InputMapping::Scan(info) = input {
                if info.axis != 0 {
                    suggestions.push((InOut::In(slot), AxisOp::Move(info.axis, 0)))
                }
            }
        }
        for output in &self.output_mapping {
            if let Some((slot, scan)) = output.scan {
                if scan.axis != 0 {
                    suggestions.push((InOut::Out(slot), AxisOp::Move(scan.axis, 0)))
                }
            }
        }
        Ok(suggestions)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        trace!("Propagating through {node}: {io:?} {change:?}");
        let body_leading_outlet = match io {
            InOut::In(ix) => self.body.input_outlets()?[ix],
            InOut::Out(slot) => {
                let output = self
                    .output_mapping
                    .iter()
                    .position(|im| {
                        im.scan.map(|(slot, _i)| slot) == Some(slot)
                            || im.last_value_slot == Some(slot)
                    })
                    .unwrap();
                self.body.output_outlets()?[output]
            }
        };
        let axis_change = AxisChange { outlet: body_leading_outlet, op: change.clone() };
        let node_input_facts = model.node_input_facts(node.id)?;
        let result = self
            .try_body_axes_change(axis_change, false, &node_input_facts)
            .with_context(|| "Attemping to run change through scan body".to_string())?;
        if result.is_some() {
            trace!("{node} accepted axis change");
        } else {
            trace!("{node} rejected axis change");
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
        pass!(declutter_const_input);
        pass!(declutter_discard_unused_input);
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

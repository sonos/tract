use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::scan::{InputMapping, Scan};

register_all!(Scan: pulsify);

fn pulsify(
    op: &Scan,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {


/*

    dbg!(source.node_axes_mapping(node.id)?.to_string());
    for input_id in &node.inputs {
        dbg!(target.outlet_fact(mapping[input_id]))?;
    }
    for input_id in 0..node.inputs.len() {
        let input = mapping[&node.inputs[input_id]];
        let input_fact = target.outlet_fact(input)?;
        if let Some(info) = op.input_mapping[input_id].as_scan() {
            if info.chunk < 0 {
                bail!("Can not pulsify a backward scan.")
            }
            if input_fact.stream.as_ref().context("scan on non-streamed input")?.axis != info.axis {
                bail!("Scan pulsification limited to scanning axis");
            }
        }
    }
*/

    let pulse_inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();

    let axes_mapping = source.node_axes_mapping(node.id)?;
    let first_scan_slot = op.input_mapping.iter().position(InputMapping::is_scan).unwrap();
    let first_scan_axis = target.outlet_fact(pulse_inputs[first_scan_slot])?.stream.as_ref().unwrap().axis;
    let scan_axis = axes_mapping.axis((InOut::In(first_scan_slot), first_scan_axis))?;
    if first_scan_axis == op.input_mapping[first_scan_slot].as_scan().unwrap().axis {
        let mut op = op.clone();
        op.skip = target.outlet_fact(pulse_inputs[first_scan_slot])?.stream.as_ref().unwrap().delay;
        for om in op.output_mapping.iter_mut() {
            if om.scan.is_some() {
                om.full_dim_hint = None;
            }
        }
        Ok(Some(target.wire_node(&*node.name, op, &pulse_inputs)?))
    } else if scan_axis.outputs.iter().all(|x| x.len() == 1) {
        let body = PulsedModel::new(&op.body, symbol.clone(), pulse)?.into_typed()?;
        let mut new_op = Scan::new(body, op.input_mapping.clone(), op.output_mapping.clone(), 0)?;
        new_op.reset_every_turn = true;
        target.wire_node(&node.name, new_op, &pulse_inputs).map(Some)
    } else {
        todo!("Unsupported pulsification")
    }
}

impl PulsedOp for Scan {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let outer_output_count = self
            .output_mapping
            .iter()
            .map(|om| om.scan.map(|s| s.0).unwrap_or(0).max(om.last_value_slot.unwrap_or(0)))
            .max()
            .context("no output?")?
            + 1;

        let first_scan_slot = self.input_mapping.iter().position(InputMapping::is_scan).unwrap();
        let first_pulse_axis = inputs[first_scan_slot].stream.as_ref().unwrap().axis;
        let first_scan_axis = self.input_mapping[first_scan_slot].as_scan().as_ref().unwrap().axis;
        let tracking = self.body.axes_mapping()?;
        let pulse_axis = tracking.axis((InOut::In(first_scan_slot), first_pulse_axis))?;
        let mut facts = tvec!();
        for output_slot in 0..outer_output_count {
            let (output_body_ix, output_mapping) = self
                .output_mapping
                .iter()
                .enumerate()
                .find(|(_ix, om)| om.scan.map(|s| s.0) == Some(output_slot))
                .context("Scan pulse only supports full outputs")?;
            let output_body_fact = self.body.output_fact(output_body_ix)?;
            let fact = if first_scan_axis == first_pulse_axis {
                let shape: ShapeFact = output_body_fact
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(axis, d)| {
                        if axis == output_mapping.scan.unwrap().1.axis {
                            inputs[first_scan_slot].pulse().unwrap().to_dim()
                        } else {
                            d.clone()
                        }
                    })
                    .collect();
                PulsedFact {
                    datum_type: output_body_fact.datum_type,
                    shape,
                    stream: Some(StreamInfo {
                        axis: output_mapping.scan.unwrap().1.axis,
                        dim: inputs[first_scan_slot].stream.as_ref().unwrap().dim.clone(),
                        delay: inputs[first_scan_slot].stream.as_ref().unwrap().delay,
                    }),
                }
            } else {
                let pulse_axis = pulse_axis.outputs[output_body_ix][0];
                let mut shape = output_body_fact.shape.clone();
                if let Some(info) = output_mapping.scan {
                    shape.set(info.0, inputs[first_scan_slot].shape[first_scan_axis].clone());
                }
                PulsedFact {
                    datum_type: output_body_fact.datum_type,
                    shape,
                    stream: Some(StreamInfo {
                        axis: pulse_axis,
                        dim: inputs[first_scan_slot].stream.as_ref().unwrap().dim.clone(),
                        delay: inputs[first_scan_slot].stream.as_ref().unwrap().delay,
                    }),
                }
            };
            facts.push(fact);
        }
        Ok(facts)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

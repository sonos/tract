use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::scan::{InputMapping, Scan};

register_all!(Scan: pulsify);

fn pulsify(
    op: &Scan,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    for input_id in 0..node.inputs.len() {
        let input = mapping[&node.inputs[input_id]];
        let input_fact = target.outlet_fact(input)?;
        let info = op
            .input_mapping
            .iter()
            .filter_map(InputMapping::as_scan)
            .find(|mapping| mapping.slot == input_id)
            .context("Scan with constant input: this is unexpected. Is the network decluttered ?")?;
        if info.chunk < 0 {
            bail!("Can not pulsify a backward scan.")
        }
        if input_fact.stream.as_ref().context("scan on non-streamed input")?.axis != info.axis {
            bail!("Scan pulsification limited to scanning axis");
        }
    }

    let pulse_inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();

    let mut op = op.clone();
    op.skip = target.outlet_fact(pulse_inputs[0])?.stream.as_ref().unwrap().delay;
    for mut om in op.output_mapping.iter_mut() {
        if om.scan.is_some() {
            om.full_dim_hint = None;
        }
    }
    Ok(Some(target.wire_node(&*node.name, op, &pulse_inputs)?))
}

impl PulsedOp for Scan {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let output_count = self
            .output_mapping
            .iter()
            .map(|om| om.scan.map(|s| s.slot).unwrap_or(0).max(om.last_value_slot.unwrap_or(0)))
            .max()
            .context("no output?")?
            + 1;

        let mut facts = tvec!();
        for output_slot in 0..output_count {
            let (output_body_ix, output_mapping) = self
                .output_mapping
                .iter()
                .enumerate()
                .find(|(_ix, om)| om.scan.map(|s| s.slot) == Some(output_slot))
                .context("Scan pulse only supports full outputs")?;
            let output_body_fact = self.body.output_fact(output_body_ix)?;
            let shape: ShapeFact = output_body_fact
                .shape
                .iter()
                .enumerate()
                .map(|(axis, d)| {
                    if axis == output_mapping.scan.unwrap().axis {
                        inputs[0].pulse().unwrap().to_dim()
                    } else {
                        d
                    }
                })
                .collect();
            let fact = PulsedFact {
                datum_type: output_body_fact.datum_type,
                shape,
                stream: Some(StreamInfo {
                    axis: output_mapping.scan.unwrap().axis,
                    dim: inputs[0].stream.as_ref().unwrap().dim.clone(),
                    delay: inputs[0].stream.as_ref().unwrap().delay,
                }),
            };
            facts.push(fact);
        }
        Ok(facts)
    }

    as_op!();
    pulsed_op_to_typed_op!();
}

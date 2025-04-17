use crate::internal::*;

use crate::ops::array::{Slice, TypedConcat};
use crate::ops::einsum::EinSum;
use crate::ops::math::add;
use crate::optim::OptimizerSession;
use tract_itertools::Itertools;

#[derive(Clone, Debug, Default)]
pub struct ConcatThenEinsum(Option<InletId>);

impl super::TypedPass for ConcatThenEinsum {
    fn reset(&mut self) -> TractResult<()> {
        self.0 = Default::default();
        Ok(())
    }

    #[allow(clippy::comparison_chain)]
    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        'outer: loop {
            self.0 = if let Some(previous) = self.0 {
                if let Some(next) = next_inlet(model, &previous) {
                    Some(next)
                } else {
                    return Ok(None);
                }
            } else if let Some(first) =
                model.nodes.iter().find(|n| n.inputs.len() > 0).map(|n| InletId::new(n.id, 0))
            {
                Some(first)
            } else {
                return Ok(None);
            };
            let inlet = self.0.unwrap();
            let outlet = model.nodes[inlet.node].inputs[inlet.slot];
            let concat_node = model.node(outlet.node);
            if model.outputs.contains(&concat_node.id.into()) {
                continue;
            }
            let einsum_node = &model.nodes[inlet.node];
            if einsum_node.inputs.len() != 2 {
                // should we try and apply this on quantized einsums ?
                continue;
            }
            if let (Some(concat), Some(einsum)) =
                (concat_node.op_as::<TypedConcat>(), einsum_node.op_as::<EinSum>())
            {
                let offsets = concat.offsets(&model.node_input_facts(concat_node.id)?)?;
                let axis_info = einsum.axes.axis((InOut::In(inlet.slot), concat.axis))?;
                // only split if axis is a summing axis
                if axis_info.outputs[0].len() > 0 {
                    continue;
                }
                let mut patch = TypedModelPatch::new(format!(
                    "Split Einsum for concat on axis {}",
                    axis_info.repr
                ));
                // inputs[einsum_input_slot][concated_slice]. concated_slice = 0 for broadcast
                let mut inputs: TVec<TVec<OutletId>> = tvec!();
                for (slot, input) in einsum_node.inputs.iter().enumerate() {
                    let tap = patch.tap_model(model, *input)?;
                    if axis_info.inputs[slot].len() > 1 {
                        continue 'outer;
                    } else if axis_info.inputs[slot].len() == 1 {
                        let mut slices = tvec!();
                        for (start, end) in offsets.iter().cloned().tuple_windows() {
                            let wire = patch.wire_node(
                                format!(
                                    "{}.concat-einsum-slice-{}.{}.{}..{}",
                                    einsum_node.name, axis_info.repr, slot, start, end
                                ),
                                Slice { axis: axis_info.inputs[slot][0], start, end },
                                &[tap],
                            )?;
                            slices.push(wire[0]);
                        }
                        inputs.push(slices);
                    } else {
                        inputs.push(tvec!(tap)); // broadcast
                    };
                }
                let mut einsums = tvec!();
                for (ix, (start, end)) in offsets.iter().tuple_windows().enumerate() {
                    let mut einsum_inputs = tvec!();
                    for input_ix in 0..einsum_node.inputs.len() {
                        einsum_inputs
                            .push(inputs[input_ix].get(ix).cloned().unwrap_or(inputs[input_ix][0]));
                    }
                    let einsum = patch.wire_node(
                        format!(
                            "{}.concat-einsum-{}.{}..{}",
                            einsum_node.name, axis_info.repr, start, end
                        ),
                        einsum.clone(),
                        &einsum_inputs,
                    )?[0];
                    einsums.push(einsum);
                }
                let wire = if let Some(axis) = axis_info.outputs[0].first().cloned() {
                    patch.wire_node(
                        format!("{}.concat-einsum-{}.concat", einsum_node.name, axis_info.repr),
                        TypedConcat { axis },
                        &einsums,
                    )?[0]
                } else {
                    let mut wire = einsums[0];
                    for ix in 1..einsums.len() {
                        wire = patch.wire_node(
                            format!(
                                "{}.concat-einsum-{}.add-{}",
                                einsum_node.name, axis_info.repr, ix
                            ),
                            add(),
                            &[wire, einsums[ix]],
                        )?[0]
                    }
                    wire
                };
                patch.shunt_outside(model, einsum_node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
    }
}

fn next_inlet(model: &TypedModel, inlet: &InletId) -> Option<InletId> {
    if inlet.slot + 1 < model.nodes[inlet.node].inputs.len() {
        Some(InletId::new(inlet.node, inlet.slot + 1))
    } else {
        model.nodes[inlet.node + 1..]
            .iter()
            .find(|n| n.inputs.len() > 0)
            .map(|n| InletId::new(n.id, 0))
    }
}

use crate::internal::*;
use crate::ops::array::Slice;
use crate::optim::OptimizerSession;

#[derive(Clone, Debug)]
pub struct PushSliceUp;

impl super::TypedPass for PushSliceUp {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(
        &mut self,
        session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        for n in model.eval_order()? {
            let (ifacts, ofacts) = model.node_facts(n)?;
            if ofacts.len() != 1 {
                continue;
            }
            let node = model.node(n);
            let invariants = node.op.invariants(&ifacts, &ofacts)?;
            'axis: for axis in 0..ofacts[0].rank() {
                if let Some(boundaries) = should_slice_output(model, &node, axis)? {
                    let mut splits = tvec!();
                    let mut patch = TypedModelPatch::new("push slice up");
                    let inputs = node
                        .inputs
                        .iter()
                        .map(|i| patch.tap_model(model, *i))
                        .collect::<TractResult<TVec<OutletId>>>()?;
                    let mut start = 0;
                    let axis_info = invariants.track_output_axis(0, axis);
                    for end in &boundaries {
                        let mut wires = tvec!();
                        for input_ix in 0..inputs.len() {
                            let mut wire = inputs[input_ix];
                            if let Some(input_axis) = axis_info.and_then(|it| it.inputs[input_ix]) {
                                wire = patch.wire_node(
                                    format!(
                                        "{}.split-{}-over-{}.{}..{}.slice",
                                        &node.name, input_ix, input_axis, start, end
                                    ),
                                    Slice {
                                        axis: input_axis,
                                        start: start.to_dim(),
                                        end: end.to_dim(),
                                    },
                                    &[wire],
                                )?[0];
                            }
                            wires.push(wire);
                        }
                        let Some(wire) = node.op.slice(
                            &mut patch,
                            &format!(
                                "{}.split-over-{}.{}..{}",
                                &node.name, axis, start, end
                                ),
                                &wires,
                                axis,
                                start,
                                *end,
                                )? else {
                            continue 'axis };
                        splits.push(wire[0]);
                        start = *end;
                    }
                    rewire_sliced_outputs(model, node, axis, &mut patch, &boundaries, &splits)?;
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }
}

pub fn should_slice_output(
    model: &TypedModel,
    node: &TypedNode,
    axis: usize,
) -> TractResult<Option<TVec<usize>>> {
    let Some(slice) = node.outputs[0].successors.iter().find_map(|inlet| {
        model.node(inlet.node).op_as::<Slice>().filter(|slice| slice.axis == axis).map(|_| inlet.node)
    }) else {
        return Ok(None)
    };
    let slice_op = model.node(slice).op_as::<Slice>().unwrap();
    let axis = slice_op.axis;
    let mut boundaries = tvec!();
    for succ in &node.outputs[0].successors {
        if let Some(slice) = model.node(succ.node).op_as::<Slice>() {
            if slice.axis == axis {
                boundaries.push(slice.start.clone());
                boundaries.push(slice.end.clone());
            }
        }
    }
    let mut boundaries: TVec<usize> = if let Ok(boundaries) =
        boundaries.iter().map(|x| x.to_usize()).collect::<TractResult<TVec<_>>>()
    {
        boundaries
    } else {
        return Ok(None);
    };
    let end = if let Ok(x) = node.outputs[0].fact.shape[axis].to_usize() {
        x
    } else {
        return Ok(None);
    };
    boundaries.push(end);
    boundaries.retain(|x| *x > 0);
    boundaries.sort();
    boundaries.dedup();
    Ok(Some(boundaries))
}

pub fn rewire_sliced_outputs(
    model: &TypedModel,
    node: &TypedNode,
    axis: usize,
    patch: &mut TypedModelPatch,
    boundaries: &[usize],
    splits: &[OutletId],
) -> TractResult<()> {
    let full = patch.wire_node(
        format!("{}.concat-{}", node.name, axis),
        crate::ops::array::TypedConcat::concat_vars(axis, splits.len()),
        &splits,
    )?[0];
    patch.shunt_outside(model, node.id.into(), full)?;
    for (ix, succ) in node.outputs[0].successors.iter().enumerate() {
        if let Some(slice) =
            model.node(succ.node).op_as::<Slice>().filter(|slice| slice.axis == axis)
        {
            // example: boundaries: 2, 3, wanted: 0..2 -> [0]
            let slices: TVec<OutletId> = boundaries
                .iter()
                .zip(splits.iter())
                .filter_map(|(up, split)| {
                    if *up > slice.start.to_usize().unwrap() && *up <= slice.end.to_usize().unwrap()
                    {
                        Some(*split)
                    } else {
                        None
                    }
                })
                .collect();
            let wire = if slices.len() > 1 {
                patch.wire_node(
                    format!("{}.concat-m{}..{}..{}", node.name, ix, slice.start, slice.end),
                    crate::ops::array::TypedConcat::concat_vars(axis, slices.len()),
                    &slices,
                )?[0]
            } else {
                slices[0]
            };
            patch.shunt_outside(model, succ.node.into(), wire)?;
        }
    }
    Ok(())
}

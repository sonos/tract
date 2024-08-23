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
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let eval_order = model.eval_order()?;
        for &n in &eval_order {
            let (ifacts, ofacts) = model.node_facts(n)?;
            if ofacts.len() != 1 {
                continue;
            }
            let node = model.node(n);
            let invariants = node
                .op
                .axes_mapping(&ifacts, &ofacts)
                .with_context(|| format!("Mapping axes for {node}"))?;
            'axis: for axis in 0..ofacts[0].rank() {
                if let Some(boundaries) = should_slice_output(model, node, axis, &eval_order)? {
                    let mut splits = tvec!();
                    let mut patch = TypedModelPatch::new(format!("Slice {node} by {boundaries:?}"));
                    let inputs = patch.taps(model, &node.inputs)?;
                    let mut start = 0;
                    let axis_info = invariants.axis((InOut::Out(0), axis)).unwrap();
                    for end in &boundaries {
                        let mut wires = tvec!();
                        for input_ix in 0..inputs.len() {
                            let mut wire = inputs[input_ix];
                            if let &[input_axis] = &*axis_info.inputs[input_ix] {
                                // dont propagate slice up if input looks like a broadcasting input
                                if !patch.outlet_fact(wire)?.shape[input_axis].is_one() {
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
                            }
                            wires.push(wire);
                        }
                        let Some(wire) = node
                            .op
                            .slice(
                                &mut patch,
                                model,
                                node,
                                &format!("{}.split-over-{}.{}..{}", &node.name, axis, start, end),
                                &wires,
                                axis,
                                start,
                                *end,
                            )
                            .with_context(|| format!("Calling slice on {node}"))?
                        else {
                            continue 'axis;
                        };
                        splits.push(wire[0]);
                        start = *end;
                    }
                    rewire_sliced_outputs(model, node, axis, &mut patch, &boundaries, &splits)
                        .context("Rewiring sliced outputs")?;
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }
}

fn should_slice_output(
    model: &TypedModel,
    node: &TypedNode,
    axis: usize,
    eval_order: &[usize],
) -> TractResult<Option<TVec<usize>>> {
    if node.outputs[0].successors.len() == 0 {
        return Ok(None);
    }
    let slicers: TVec<usize> = node.outputs[0]
        .successors
        .iter()
        .filter(|inlet| {
            model.node(inlet.node).op_as::<Slice>().filter(|slice| slice.axis == axis).is_some()
        })
        .map(|inlet| inlet.node)
        .collect();
    /* aggressive: 1 slice as succesor => we propagate it */
    /*
    let Some(slice) = node.outputs[0].successors.iter().find_map(|inlet| {
        model.node(inlet.node).op_as::<Slice>().filter(|slice| slice.axis == axis).map(|_| inlet.node)
    }) else {
        return Ok(None)
    };
    */
    /* non-aggressive: we need all consumers to be slice */
    if slicers.len() < node.outputs[0].successors.len() {
        return Ok(None);
    }
    let slice = node.outputs[0].successors[0].node;

    if !eval_order.contains(&slice) {
        return Ok(None);
    }
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
    if boundaries.len() == 0 || (boundaries.len() == 1 && boundaries[0] == end) {
        Ok(None)
    } else {
        Ok(Some(boundaries))
    }
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
        crate::ops::array::TypedConcat::new(axis),
        splits,
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
                    crate::ops::array::TypedConcat::new(axis),
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

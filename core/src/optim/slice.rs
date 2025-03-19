use tract_itertools::Itertools;

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
            let node = model.node(n);
            if model.node(n).outputs.len() != 1 {
                continue;
            }
            for axis in 0..node.outputs[0].fact.rank() {
                if let Some(succ) = model.single_succ(n)? {
                    let Some(slice) = succ.op_as::<Slice>() else { continue };
                    let full_len = &node.outputs[0].fact.shape[axis];
                    if slice.axis != axis {
                        continue;
                    }
                    if let Some(me) = node.op_as::<Slice>() {
                        let my_len = &node.outputs[0].fact.shape[me.axis];
                        let slice_len = &succ.outputs[0].fact.shape[slice.axis];
                        if !(my_len.clone() - slice_len).prove_strict_positive() {
                            continue;
                        }
                    }
                    let boundaries =
                        tvec!(0.to_dim(), slice.start.clone(), slice.end.clone(), full_len.clone());
                    let Some((mut patch, splits)) =
                        op_slices_to_slice_op(model, node, axis, &boundaries)?
                    else {
                        continue;
                    };
                    ensure!(splits.len() == 3);
                    // ignore first split (0..start)
                    let wire = splits[1];
                    patch.shunt_outside(model, succ.id.into(), wire)?;

                    return Ok(Some(patch));
                // handle multiple slicing successors in fan-out fashion (think LSTM post linear op)
                // limited to concrete interger slicing boundaries for ordering
                // (it may actually work with generic TDim with ordering)
                } else if let Some(boundaries) =
                    should_slice_output(model, node, axis, &eval_order)?
                {
                    let boundaries_dim: TVec<TDim> =
                        boundaries.iter().map(|d| d.to_dim()).collect();
                    let Some((mut patch, splits)) =
                        op_slices_to_slice_op(model, node, axis, &boundaries_dim)?
                    else {
                        continue;
                    };
                    ensure!(splits.len() == boundaries.len() - 1);
                    rewire_sliced_outputs(model, node, axis, &mut patch, &boundaries, &splits)
                        .context("Rewiring sliced outputs")?;
                    return Ok(Some(patch));
                }
            }
        }
        Ok(None)
    }
}

fn op_slices_to_slice_op(
    model: &TypedModel,
    node: &TypedNode,
    axis: usize,
    boundaries: &[TDim],
) -> TractResult<Option<(TypedModelPatch, TVec<OutletId>)>> {
    let (ifacts, ofacts) = model.node_facts(node.id)?;
    let invariants = node
        .op
        .axes_mapping(&ifacts, &ofacts)
        .with_context(|| format!("Mapping axes for {node}"))?;
    let mut splits = tvec!();
    let mut patch = TypedModelPatch::new(format!("Slice {node} by {boundaries:?}"));
    let inputs = patch.taps(model, &node.inputs)?;
    let len = &node.outputs[0].fact.shape[axis];
    ensure!(boundaries[0] == 0.to_dim());
    ensure!(boundaries.last().as_ref().unwrap() == &len);
    let axis_info = invariants.axis((InOut::Out(0), axis)).unwrap();
    for (start, end) in boundaries.iter().tuple_windows() {
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
                        Slice { axis: input_axis, start: start.to_dim(), end: end.to_dim() },
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
                end,
            )
            .with_context(|| format!("Calling slice on {node}"))?
        else {
            return Ok(None);
        };
        splits.push(wire[0]);
    }
    Ok(Some((patch, splits)))
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
    if node.op_is::<Slice>() {
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
    boundaries.sort();
    boundaries.dedup();
    if boundaries.len() == 2 {
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
    let zero = patch.add_const(
        format!("{}.zero", node.name),
        Tensor::zero_scalar_dt(node.outputs[0].fact.datum_type)?,
    )?;
    for (ix, succ) in node.outputs[0].successors.iter().enumerate() {
        if let Some(slice) =
            model.node(succ.node).op_as::<Slice>().filter(|slice| slice.axis == axis)
        {
            // example: boundaries: 2, 3, wanted: 0..2 -> [0]
            let slices: TVec<OutletId> = boundaries
                .iter()
                .tuple_windows()
                .zip(splits.iter())
                .filter_map(|((_down, up), split)| {
                    if *up > slice.start.to_usize().unwrap() && *up <= slice.end.to_usize().unwrap()
                    {
                        Some(*split)
                    } else {
                        None
                    }
                })
                .collect();
            let wire = if slices.len() == 0 {
                let mut empty_shape = node.outputs[0].fact.shape.clone();
                empty_shape.set(axis, 0.to_dim());
                patch.wire_node(
                    format!("{}.concat-m{}..{}..{}", node.name, ix, slice.start, slice.end),
                    crate::ops::array::MultiBroadcastTo::new(empty_shape),
                    &[zero],
                )?[0]
            } else if slices.len() > 1 {
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

use tract_data::itertools::izip;

use crate::broadcast::multi_broadcast;
use crate::internal::*;
use crate::ops::binary::TypedBinOp;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct MultiBroadcastTo {
    pub shape: ShapeFact,
}

impl Op for MultiBroadcastTo {
    fn name(&self) -> StaticName {
        "MultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MultiBroadcastTo {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let shape = self.shape.eval_to_usize(ctx.symbols)?;
        if inputs[0].shape() == shape.as_slice() {
            return Ok(tvec!(inputs[0].clone()));
        }
        if let Some(out) = fast_broadcast(&inputs[0], &shape)? {
            return Ok(tvec!(out.into_tvalue()));
        }
        Ok(tvec!(inputs[0].broadcast_to_shape(&shape)?.into_tvalue()))
    }
}

impl TypedOp for MultiBroadcastTo {
    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        // ONNX-style broadcasting right-aligns input over output, so when
        // output_rank > input_rank the leading output axes are pure
        // broadcast axes with no input correspondence. natural_for_rank's
        // square shape would skip them and trip the optimizer's axes-mapping
        // check (caught under paranoid_assertions).
        let in_rank = inputs[0].rank();
        let out_rank = outputs[0].rank();
        let leading = out_rank.saturating_sub(in_rank);
        let mut axes = tvec!();
        let mut alphabet = 'a'..;
        for o in 0..leading {
            axes.push(
                Axis::new(alphabet.next().unwrap(), inputs.len(), outputs.len()).output(0, o),
            );
        }
        for i in 0..in_rank.min(out_rank) {
            axes.push(
                Axis::new(alphabet.next().unwrap(), inputs.len(), outputs.len())
                    .input(0, i)
                    .output(0, leading + i),
            );
        }
        AxesMapping::new(inputs.len(), outputs.len(), axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        // Only propagate axis changes that touch passthrough axes — those
        // where the input and output shapes agree. Touching a broadcast
        // axis (input=1, output=N) would make the input and output rank
        // diverge through the change and break the broadcast relationship,
        // and propagating Rm of a non-trivial axis into a Source produces
        // the "Removing non-trivial axis" hard error from change_shape.
        let input_shape = &model.outlet_fact(node.inputs[0])?.shape;
        let canonical = change.canonical();
        let touched: TVec<usize> = match canonical.as_ref() {
            AxisOp::Add(ix) | AxisOp::Rm(ix) => tvec![*ix],
            AxisOp::Move(from, to) => {
                rule_if!(input_shape.rank() == self.shape.rank());
                tvec![*from, *to]
            }
            _ => return Ok(None),
        };
        for &ix in &touched {
            if ix < self.shape.rank()
                && ix < input_shape.rank()
                && input_shape[ix] != self.shape[ix]
            {
                return Ok(None);
            }
        }

        let mut shape = self.shape.clone();
        if change.change_shape(&mut shape, false).is_ok() {
            return Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(MultiBroadcastTo { shape })),
                change,
            )));
        }
        Ok(None)
    }

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let mut fact = inputs[0].datum_type.fact(self.shape.clone());
        fact.uniform.clone_from(&inputs[0].uniform);
        fact.uniform_tdim = inputs[0].uniform_tdim.clone();
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        crate::optim::propagate_roi::bubble_roi(model, node)
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let shape: TVec<_> =
            self.shape.iter().map(|d| d.substitute_all(subs)).collect::<TractResult<_>>()?;
        let op = Self { shape: shape.into() };
        target.wire_node(&node.name, op, &[input])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape == self.shape {
            return TypedModelPatch::shunt_one_op(model, node);
        }
        // Swap with an AxisOp successor: `Broadcast(x, S) → AxisOp` becomes
        // `AxisOp(x) → Broadcast(σ(S))` whenever the AxisOp transforms every
        // axis the broadcast actually expanded.  Fires per-successor, so this
        // works under fan-out (the original broadcast stays in place for
        // siblings; only the matched AxisOp branch is rerouted).
        for succ in &*node.outputs[0].successors {
            let succ = model.node(succ.node);
            let Some(op) = succ.op_as::<AxisOp>() else { continue };
            // The AxisOp's indices refer to the broadcast output; they are only
            // meaningful on the input if the broadcast did not add leading axes.
            if input_fact.rank() != self.shape.rank() {
                continue;
            }
            let mut shape = self.shape.clone();
            if izip!(0.., &*input_fact.shape, &*self.shape)
                .filter(|(_, l, r)| l != r)
                .all(|(axis, _, _)| op.transform_axis(axis).is_some())
                && op.change_shape(&mut shape, false).is_ok()
            {
                let mut patch = TypedModelPatch::default();
                let mut wire = patch.tap_model(model, node.inputs[0])?;
                wire = patch.wire_node(&succ.name, op.clone(), &[wire])?[0];
                wire = patch.wire_node(&node.name, MultiBroadcastTo { shape }, &[wire])?[0];
                patch.shunt_outside(model, succ.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
        if let [succ] = &*node.outputs[0].successors {
            let succ = model.node(succ.node);
            if succ.op_is::<TypedBinOp>() {
                let our_slot = node.outputs[0].successors[0].slot;
                let other_slot = 1 - our_slot;
                let other_operand = succ.inputs[other_slot];
                let other_fact = model.outlet_fact(other_operand)?;
                let output_fact = model.outlet_fact(succ.id.into())?;
                if input_fact.rank() == other_fact.rank()
                    && multi_broadcast(&[&input_fact.shape, &other_fact.shape])
                        .is_ok_and(|s| *s == *output_fact.shape)
                {
                    let mut operands = tvec!(node.inputs[0], other_operand);
                    if our_slot == 1 {
                        operands.swap(0, 1);
                    }
                    return TypedModelPatch::rewire(
                        model,
                        &operands,
                        &[succ.id.into()],
                        &|p, inputs| p.wire_node(&succ.name, succ.op.clone(), inputs),
                    )
                    .map(Some);
                }
            }
        }
        Ok(None)
    }

    as_op!();
}

/// Copy-typed broadcast: repeat inner blocks instead of ndarray's general
/// strided `into_owned`. Nearest upsample (Tile of inserted 1-axes) and
/// channel-bias broadcast to NCHW hit this path.
fn fast_broadcast(input: &Tensor, out_shape: &[usize]) -> TractResult<Option<Tensor>> {
    if !input.datum_type().is_copy() {
        return Ok(None);
    }
    if input.rank() > out_shape.len() {
        return Ok(None);
    }
    if input.len() != input.shape().iter().product::<usize>()
        || input.strides() != &*Tensor::natural_strides(input.shape())
    {
        return Ok(None);
    }
    let rank = out_shape.len();
    let mut in_shape = vec![1usize; rank];
    in_shape[rank - input.rank()..].copy_from_slice(input.shape());
    for (&i, &o) in in_shape.iter().zip(out_shape.iter()) {
        if i != 1 && i != o {
            return Ok(None);
        }
    }
    let mut output = unsafe { Tensor::uninitialized_dt(input.datum_type(), out_shape)? };
    if output.len() == 0 {
        return Ok(Some(output));
    }
    let item = input.datum_type().size_of();
    unsafe {
        broadcast_copy(
            input.as_bytes().as_ptr(),
            &in_shape,
            output.as_bytes_mut().as_mut_ptr(),
            out_shape,
            item,
        );
    }
    Ok(Some(output))
}

unsafe fn broadcast_copy(
    inp: *const u8,
    in_shape: &[usize],
    out: *mut u8,
    out_shape: &[usize],
    item: usize,
) {
    unsafe fn fill(
        axis: usize,
        inp: *const u8,
        out: *mut u8,
        in_shape: &[usize],
        out_shape: &[usize],
        item: usize,
    ) {
        unsafe {
            let rank = out_shape.len();
            if axis == rank - 1 {
                let nout = out_shape[axis];
                if in_shape[axis] == nout {
                    std::ptr::copy_nonoverlapping(inp, out, nout * item);
                } else {
                    debug_assert_eq!(in_shape[axis], 1);
                    if item == 4 {
                        let v = *(inp as *const u32);
                        let dst = out as *mut u32;
                        for i in 0..nout {
                            *dst.add(i) = v;
                        }
                    } else {
                        for i in 0..nout {
                            std::ptr::copy_nonoverlapping(inp, out.add(i * item), item);
                        }
                    }
                }
                return;
            }
            let inner_in: usize = in_shape[axis + 1..].iter().product::<usize>() * item;
            let inner_out: usize = out_shape[axis + 1..].iter().product::<usize>() * item;
            if in_shape[axis] == 1 {
                fill(axis + 1, inp, out, in_shape, out_shape, item);
                for i in 1..out_shape[axis] {
                    std::ptr::copy_nonoverlapping(out, out.add(i * inner_out), inner_out);
                }
            } else {
                for i in 0..in_shape[axis] {
                    fill(
                        axis + 1,
                        inp.add(i * inner_in),
                        out.add(i * inner_out),
                        in_shape,
                        out_shape,
                        item,
                    );
                }
            }
        }
    }
    unsafe {
        fill(0, inp, out, in_shape, out_shape, item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::change_axes::AxisOp;
    use crate::ops::logic::And;

    /// `Broadcast → Move` with the broadcast feeding a SINGLE successor.
    /// Pre-existing path: the swap rewrite kicks in.
    #[test]
    fn broadcast_move_single_successor_swaps() -> TractResult<()> {
        let mut model = TypedModel::default();
        let t = model.symbols.sym("T");
        let pad = model.add_source("pad", bool::fact(&[t.to_dim()]))?;
        let unsq = model.wire_node("unsq", AxisOp::Add(0), &[pad])?[0];
        let bcast = model.wire_node(
            "bcast",
            MultiBroadcastTo { shape: ShapeFact::from_dims([t.to_dim(), t.to_dim()]) },
            &[unsq],
        )?[0];
        let mv = model.wire_node("move", AxisOp::Move(0, 1), &[bcast])?[0];
        model.select_output_outlets(&[mv])?;

        let model = model.into_decluttered()?;

        let move_count = model
            .nodes()
            .iter()
            .filter(|n| matches!(n.op_as::<AxisOp>(), Some(AxisOp::Move(0, 1))))
            .count();
        assert_eq!(move_count, 0, "Move should have been pushed through Broadcast and absorbed");
        Ok(())
    }

    /// `Broadcast → {Move, And-direct}` — the encoder-style pad-mask outer-AND
    /// pattern.  Pre-fix: declutter bailed because broadcast had > 1 successor;
    /// the Move stayed.  Post-fix: the Move-branch gets its own swapped
    /// chain, the direct-AND branch still consumes the original broadcast.
    #[test]
    fn broadcast_move_fanout_pushes_through_one_branch() -> TractResult<()> {
        let mut model = TypedModel::default();
        let t = model.symbols.sym("T");
        let pad = model.add_source("pad", bool::fact(&[t.to_dim()]))?;
        let unsq = model.wire_node("unsq", AxisOp::Add(0), &[pad])?[0];
        let bcast = model.wire_node(
            "bcast",
            MultiBroadcastTo { shape: ShapeFact::from_dims([t.to_dim(), t.to_dim()]) },
            &[unsq],
        )?[0];
        let mv = model.wire_node("move", AxisOp::Move(0, 1), &[bcast])?[0];
        let and = model.wire_node("and", TypedBinOp(Box::new(And), None), &[bcast, mv])?[0];
        model.select_output_outlets(&[and])?;

        let model = model.into_decluttered()?;

        // Expected: fan-out swap-through fires on the Move branch, then the
        // existing Broadcast→TypedBinOp rule fires on each (now single-
        // successor) broadcast, eliminating both — the AND ends up
        // broadcasting [1, T] and [T, 1] implicitly.
        let bcast_count = model.nodes().iter().filter(|n| n.op_is::<MultiBroadcastTo>()).count();
        assert_eq!(
            bcast_count, 0,
            "Both broadcasts should be subsumed into AND's implicit broadcasting"
        );

        let and_node =
            model.nodes().iter().find(|n| n.op_is::<TypedBinOp>()).expect("AND should survive");
        assert_eq!(and_node.inputs.len(), 2);
        let and_input_shapes: Vec<_> = and_node
            .inputs
            .iter()
            .map(|i| model.outlet_fact(*i).unwrap().shape.to_tvec())
            .collect();
        let expected_a = tvec![1.to_dim(), t.to_dim()];
        let expected_b = tvec![t.to_dim(), 1.to_dim()];
        let (a, b) = (&and_input_shapes[0], &and_input_shapes[1]);
        assert!(
            (a == &expected_a && b == &expected_b) || (a == &expected_b && b == &expected_a),
            "AND should receive [1, T] and [T, 1]; got {a:?} and {b:?}"
        );
        Ok(())
    }

    /// `Broadcast → AxisOp` where the broadcast adds a leading axis (input
    /// rank < output rank).  The AxisOp's indices refer to the output shape
    /// and are meaningless on the input; the swap must not fire.  Pre-fix,
    /// the guard izip truncated to the shorter rank and wiring the AxisOp
    /// onto the input panicked in AxisOp::change_shape.
    #[test]
    fn broadcast_adding_leading_axis_does_not_swap_with_axis_op() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact([512, 1]))?;
        let bcast = model.wire_node(
            "bcast",
            MultiBroadcastTo {
                shape: ShapeFact::from_dims([1.to_dim(), 512.to_dim(), 16.to_dim()]),
            },
            &[src],
        )?[0];
        let unsq = model.wire_node("unsq", AxisOp::Add(3), &[bcast])?[0];
        model.select_output_outlets(&[unsq])?;

        let model = model.into_decluttered()?;
        assert_eq!(
            model.output_fact(0)?.shape.to_tvec(),
            tvec![1.to_dim(), 512.to_dim(), 16.to_dim(), 1.to_dim()]
        );
        Ok(())
    }

    #[test]
    fn fast_broadcast_pixel_replication_and_bias() -> TractResult<()> {
        // Nearest-upsample Tile pattern: [N,C,H,1,W,1] → [N,C,H,2,W,2].
        let src = Tensor::from_shape(
            &[1, 2, 3, 1, 4, 1],
            &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
        )?;
        let out = fast_broadcast(&src, &[1, 2, 3, 2, 4, 2])?.expect("copy path");
        let view = out.to_plain_array_view::<f32>()?;
        for c in 0..2 {
            for h in 0..3 {
                for w in 0..4 {
                    let v = (c * 12 + h * 4 + w) as f32;
                    for rh in 0..2 {
                        for rw in 0..2 {
                            assert_eq!(view[[0, c, h, rh, w, rw]], v);
                        }
                    }
                }
            }
        }
        // Channel bias [1,C,1,1] → [1,C,H,W].
        let bias = Tensor::from_shape(&[1, 3, 1, 1], &[1.0f32, 2.0, 3.0])?;
        let out = fast_broadcast(&bias, &[1, 3, 5, 6])?.expect("copy path");
        let view = out.to_plain_array_view::<f32>()?;
        for c in 0..3 {
            for h in 0..5 {
                for w in 0..6 {
                    assert_eq!(view[[0, c, h, w]], (c as f32) + 1.0);
                }
            }
        }
        Ok(())
    }
}

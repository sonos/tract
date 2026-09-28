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
        // The output always takes the change: the broadcast absorbs it in the
        // target shape. The input only takes it when every touched axis shows
        // same-index agreement between input and target shapes — a true axis
        // correspondence only when the ranks are equal (right-aligned
        // broadcasting pairs axes by index only then). An axis the input does
        // not have, or on which it is broadcast (input=1, output=N), is none
        // of the input's business — propagating the change there anyway asks a
        // rank-1 [1] wire to grow an axis it cannot express.
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
        let passthrough = touched.iter().all(|&ix| {
            ix < input_shape.rank() && ix < self.shape.rank() && input_shape[ix] == self.shape[ix]
        });
        // A Move can only ever propagate: absorbing a transpose in the target
        // re-pairs the input's dims against transposed positions, and
        // transpose-of-broadcast is NOT broadcast-into-transposed-target (a
        // [1,7] input into a transposed [7,7] target silently yields the
        // transposed tensor). Block instead, like the pre-rework guard did.
        if matches!(canonical.as_ref(), AxisOp::Move(..)) && !passthrough {
            return Ok(None);
        }

        let mut shape = self.shape.clone();
        if change.change_shape(&mut shape, false).is_ok() {
            let mut wire_changes: TVec<(InOut, AxisOp)> = tvec![(InOut::Out(0), change.clone())];
            if passthrough {
                wire_changes.push((InOut::In(0), change.clone()));
            } else {
                // The input keeps its shape while the target absorbs the
                // change. Two things must hold.
                //
                // Evaluability: the input must still right-align-broadcast
                // into the new target (removing or adding a dim-1 axis shifts
                // the pairing when ranks differ — a [5,1] input into a [9,5,1]
                // target must not become [9,5]).
                //
                // Value preservation: the input pairs with the right-aligned
                // window [offset, offset+r) of the target. A change INSIDE
                // that window shifts every input axis left of it onto a
                // neighboring target axis: when adjacent target dims are
                // equal the result still evaluates — and computes permuted
                // values ([5,1] into [5,5,1], absorbing Rm(2), yields the
                // transpose). Such a change is only harmless when the input
                // axes it would displace are all 1 (they broadcast anywhere),
                // or when it falls outside the window.
                let offset = self.shape.rank().saturating_sub(input_shape.rank());
                let r = input_shape.rank();
                let k = touched[0].saturating_sub(offset);
                let window_safe = k == 0 || k >= r || (0..k).all(|i| input_shape[i] == 1.to_dim());
                let broadcastable = multi_broadcast(&[input_shape, &shape])
                    .is_ok_and(|b| b.as_slice() == shape.as_ref());
                if !(window_safe && broadcastable) {
                    return Ok(None);
                }
            }
            return Ok(Some(AxisChangeConsequence {
                wire_changes,
                substitute_op: Some(Box::new(MultiBroadcastTo { shape })),
            }));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::change_axes::AxisOp;
    use crate::ops::change_axes::{AxisChange, InOut};
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

    /// An axis change touching an axis the input does not have is a broadcast
    /// axis: it must be absorbed in the target shape, NOT propagated to the
    /// input wire — a rank-1 [1] input cannot express it (the MMS graph hit
    /// this as "required_rank 2 vs 1" when the change reached a [1] const).
    #[test]
    fn broadcast_axis_change_absorbed_not_propagated_to_input() -> TractResult<()> {
        let mut model = TypedModel::default();
        let c = model.add_const("c", tensor1(&[0f32]))?;
        let s = model.symbols.sym("S");
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![s.to_dim(), 1usize.to_dim()])),
            &[c],
        )?[0];
        let node = model.node(y.node);
        let consequence = node
            .op
            .change_axes(&model, node, InOut::Out(0), &AxisOp::Add(2))?
            .context("expected the broadcast to absorb the change")?;
        assert_eq!(consequence.wire_changes, tvec![(InOut::Out(0), AxisOp::Add(2))]);
        Ok(())
    }

    /// Passthrough axis changes (input and output agree on the axis) still
    /// reach the input.
    #[test]
    fn passthrough_axis_change_propagates_to_input() -> TractResult<()> {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let src = model.add_source("src", f32::fact([s.to_dim(), 1usize.into()]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![s.to_dim(), 1usize.to_dim()])),
            &[src],
        )?[0];
        let node = model.node(y.node);
        let consequence = node
            .op
            .change_axes(&model, node, InOut::Out(0), &AxisOp::Add(1))?
            .context("expected propagation")?;
        assert_eq!(
            consequence.wire_changes,
            tvec![(InOut::Out(0), AxisOp::Add(1)), (InOut::In(0), AxisOp::Add(1))]
        );
        Ok(())
    }

    /// End-to-end through the ChangeAxes search: an axis change crossing a
    /// broadcast fed by a volume-1 constant must produce an applicable patch
    /// without any const rewiring.
    #[test]
    fn axis_change_through_broadcast_with_volume_one_const() -> TractResult<()> {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let definer = model.add_source("definer", f32::fact([s.to_dim()]))?;
        let c = model.add_const("c", tensor1(&[0f32]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![
                s.to_dim(),
                1usize.to_dim(),
                1usize.to_dim()
            ])),
            &[c],
        )?;
        let out = model.wire_node("out", AxisOp::Add(0), &[y[0]])?;
        model.select_output_outlets(&[out[0], definer])?;
        let change = AxisChange { outlet: out[0], op: AxisOp::Add(3) };
        let mut explored = Default::default();
        let (patch, _) =
            crate::optim::change_axes::change_axes(&model, &change, &[], &[], &mut explored)?
                .context("axis change through a broadcast should apply")?;
        patch.apply(&mut model)?;
        model.compact()?;
        let found = crate::internal::TypedSimplePlan::new(model)?
            .run(tvec!(tensor1(&[0f32; 2]).into_tvalue()))?;
        assert_eq!(found[0].shape(), &[1, 2, 1, 1, 1]);
        Ok(())
    }

    /// Removing an output axis on which the input is broadcast (input=1,
    /// output=N) is absorbed in the target; the input keeps its axis and the
    /// right-aligned broadcast still pairs it correctly.
    #[test]
    fn rm_of_broadcast_axis_absorbed() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact([5usize, 7usize]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![
                1usize.to_dim(),
                5usize.to_dim(),
                7usize.to_dim()
            ])),
            &[src],
        )?[0];
        let node = model.node(y.node);
        let consequence = node
            .op
            .change_axes(&model, node, InOut::Out(0), &AxisOp::Rm(0))?
            .context("expected the broadcast to absorb the removal")?;
        assert_eq!(consequence.wire_changes, tvec![(InOut::Out(0), AxisOp::Rm(0))]);
        let substitute = consequence
            .substitute_op
            .as_deref()
            .and_then(|op| op.as_op().downcast_ref::<MultiBroadcastTo>())
            .context("expected a MultiBroadcastTo substitute")?;
        assert_eq!(substitute.shape.to_tvec(), tvec![5.to_dim(), 7.to_dim()]);
        Ok(())
    }

    /// A Move touching an axis on which the input is broadcast must be
    /// blocked: absorbing a transpose in the target would re-pair the input's
    /// dims against transposed positions (transpose-of-broadcast is not
    /// broadcast-into-transposed-target — a [1,7] input into a transposed
    /// [7,7] target silently yields the transposed tensor).
    #[test]
    fn move_over_broadcast_axis_blocked() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact([5usize, 1usize]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![5usize.to_dim(), 7usize.to_dim()])),
            &[src],
        )?[0];
        let node = model.node(y.node);
        let blocked = node
            .op
            .change_axes(&model, node, InOut::Out(0), &AxisOp::Move(0, 1))
            .map(|r| r.is_none())
            .unwrap_or(false);
        assert!(blocked, "expected the broadcast to block the move");
        Ok(())
    }

    /// A passthrough Move (input and target agree on both touched axes) still
    /// propagates to the input.
    #[test]
    fn passthrough_move_propagates_to_input() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact([5usize, 7usize]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![5usize.to_dim(), 7usize.to_dim()])),
            &[src],
        )?[0];
        let node = model.node(y.node);
        let consequence = node
            .op
            .change_axes(&model, node, InOut::Out(0), &AxisOp::Move(0, 1))?
            .context("expected propagation")?;
        assert_eq!(
            consequence.wire_changes,
            tvec![(InOut::Out(0), AxisOp::Move(0, 1)), (InOut::In(0), AxisOp::Move(0, 1))]
        );
        Ok(())
    }

    /// Absorbing a change on the output side must keep the input
    /// right-align-broadcastable into the new target. The ChangeAxes pass
    /// proactively proposes removing any dim-1 output axis: for a [5,1] input
    /// into a [9,5,1] target, absorbing Rm(2) yields a [9,5] target the input
    /// cannot broadcast into — the change must block, and the optimized model
    /// must keep running.
    #[test]
    fn absorbed_change_keeps_input_broadcastable() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact([5usize, 1usize]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![
                9usize.to_dim(),
                5usize.to_dim(),
                1usize.to_dim()
            ])),
            &[src],
        )?[0];
        let squeezed = model.wire_node("squeezed", AxisOp::Rm(2), &[y])?[0];
        model.select_output_outlets(&[squeezed])?;
        let optimized = model.into_optimized()?;
        let found = crate::internal::TypedSimplePlan::new(optimized)?
            .run(tvec!(tensor2(&[[0f32; 1]; 5]).into_tvalue()))?;
        assert_eq!(found[0].shape(), &[9, 5]);
        Ok(())
    }

    /// An absorbed change falling inside the input-paired window re-pairs the
    /// input's axes onto neighboring target axes: with adjacent equal dims the
    /// result still evaluates but computes permuted values. A [5,1] input into
    /// a [5,5,1] target right-aligns the input's 5 onto target axis 1 (note
    /// the alignment: the input's trailing 1 pairs with the target's trailing
    /// 1), so absorbing Rm(2) would swap the pairing — the change must block
    /// and the optimized model must keep the unoptimized values.
    #[test]
    fn absorbed_change_does_not_repair_input_axes() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact([5usize, 1usize]))?;
        let y = model.wire_node(
            "y",
            MultiBroadcastTo::new(ShapeFact::from_dims(tvec![
                5usize.to_dim(),
                5usize.to_dim(),
                1usize.to_dim()
            ])),
            &[src],
        )?[0];
        let squeezed = model.wire_node("squeezed", AxisOp::Rm(2), &[y])?[0];
        model.select_output_outlets(&[squeezed])?;
        let input: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let input = Tensor::from_shape(&[5, 1], &input)?.into_tvalue();
        let raw =
            crate::internal::TypedSimplePlan::new(model.clone())?.run(tvec!(input.clone()))?;
        let optimized = model.into_optimized()?;
        let optimized_values =
            crate::internal::TypedSimplePlan::new(optimized)?.run(tvec!(input))?;
        assert_eq!(*raw[0], *optimized_values[0]);
        Ok(())
    }
}

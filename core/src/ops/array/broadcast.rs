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
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let shape = self.shape.eval_to_usize(&session.resolved_symbols)?;
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
                        .is_ok_and(|s| &*s == &*output_fact.shape)
                {
                    let mut operands = tvec!(node.inputs[0], other_operand);
                    if our_slot == 1 {
                        operands.swap(0, 1);
                    }
                    return TypedModelPatch::rewire(
                        &model,
                        &operands,
                        &[succ.id.into()],
                        &|p, inputs| p.wire_node(&succ.name, succ.op.clone(), &inputs),
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
}

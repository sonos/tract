use tract_data::itertools::izip;

use crate::broadcast::multi_broadcast;
use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::logic::Comp;

#[derive(Debug, Clone, new, Hash)]
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
        AxesMapping::natural_for_rank(inputs.len(), outputs.len(), inputs[0].rank())
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let mut shape = self.shape.clone();
        let actual_change = if io.is_input() { change.recip() } else { change.clone() };
        if actual_change.change_shape(&mut shape, false).is_ok() {
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
        Ok(tvec!(fact))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let op =
            Self { shape: self.shape.iter().map(|d| d.eval(values)).collect::<TVec<_>>().into() };
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
        if let [succ] = &*node.outputs[0].successors {
            let succ = model.node(succ.node);
            if let Some(op) = succ.op_as::<AxisOp>() {
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
            if succ.op_is::<TypedBinOp>() || succ.op_is::<Comp>() {
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

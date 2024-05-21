use tract_linalg::frame::block_quant::BlockQuant;

use crate::internal::*;
use crate::ops::einsum::codegen::{ensure_mkn_axes, AxesOrPatch};
use crate::ops::einsum::EinSum;
use crate::transform::ModelTransform;

#[derive(Debug)]
pub struct BlockQuantTransform;

impl ModelTransform for BlockQuantTransform {
    fn name(&self) -> Cow<str> {
        "BlockQuantTransform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::<()>::default()
            .with_rule_for("block_quant_einsum_weights", block_quant_einsum_weights)
            .rewrite(&(), model)
    }
}

fn block_quant_einsum_weights(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    prefix: &str,
    op: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    let &[a, b] = &*model.node_input_facts(node.id)? else { return Ok(None) };
    if b.konst.is_some() {
        let mut new_axes = op.axes.clone().with_extra_input(2)?;
        for (ix, axis) in op.axes.axes(InOut::In(0)).enumerate() {
            new_axes = new_axes.with_extra_axis_occurency(axis.repr, InOut::In(2), ix)?;
        }
        new_axes = new_axes.remove_slot(InOut::In(0))?;
        return Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &[node.inputs[1], node.inputs[0]],
            EinSum { axes: new_axes, ..op.clone() },
        )?));
    }
    if a.konst.is_none() || a.rank() != 2 {
        return Ok(None);
    }
    let AxesOrPatch::Axes(m, k, n) = ensure_mkn_axes(op, model, node)? else { return Ok(None) };
    if m.inputs[0][0] == 1 && k.inputs[0][0] == 0 {
        let a: &Tensor = a.konst.as_ref().unwrap();
        let mut patch = TypedModelPatch::default();
        let konst =
            patch.add_const(&model.node(node.inputs[0].node).name, a.clone().move_axis(1, 0)?)?;
        let axes = op
            .axes
            .clone()
            .with_extra_axis_occurency(k, InOut::In(0), 2)?
            .remove_axis_occurency(InOut::In(0), 0)?;
        let tap = patch.tap_model(model, node.inputs[1])?;
        let output = patch.wire_node(prefix, EinSum { axes, ..op.clone() }, &[konst, tap])?;
        patch.shunt_outside(model, node.id.into(), output[0])?;
        return Ok(Some(patch));
    }
    return Ok(None);
}

#[derive(Debug, Clone, Hash)]
pub struct DeBlockQuant {
    bq: Box<dyn BlockQuant>,
    fact: TypedFact,
}

impl Op for DeBlockQuant {
    fn name(&self) -> Cow<str> {
        "DeBlockQuant".into()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other
            .downcast_ref::<Self>()
            .map(|other| other.bq.same_as(&*self.bq) && other.fact == self.fact)
            .unwrap_or(false)
    }

    op_as_typed_op!();
}

impl EvalOp for DeBlockQuant {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, _inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        unreachable!()
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl OpState for DeBlockQuant {
    fn eval(
        &mut self,
        _session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let blob = input.to_scalar::<Blob>()?;
        Ok(tvec!(self.bq.dequant_f32(&blob)?.into_tvalue()))
    }
}

trivial_op_state_freeeze!(DeBlockQuant);

impl TypedOp for DeBlockQuant {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.fact.clone()))
    }
    as_op!();
}

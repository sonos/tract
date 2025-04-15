use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0};

use crate::internal::*;
use crate::ops::einsum::einsum_matmul::EinSumMatMul;
use crate::ops::konst::Const;
use crate::transform::ModelTransform;

#[derive(Debug)]
pub struct BlockQuantTransform;

impl ModelTransform for BlockQuantTransform {
    fn name(&self) -> Cow<str> {
        "BlockQuantTransform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        crate::ops::einsum::einsum_matmul::detect_all(model)?;
        Rewriter::<()>::default()
            .with_rule_for("block_quant_einsum_weights", block_quant_einsum_weights)
            .rewrite(&(), model)?;
        crate::ops::einsum::einsum_matmul::flatten_all(model)?;
        Ok(())
    }
}

fn block_quant_einsum_weights(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    prefix: &str,
    op: &EinSumMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 2 {
        return Ok(None);
    }
    for (slot, fact) in model.node_input_facts(node.id)?.iter().enumerate() {
        let Some(a) = fact.konst.as_ref() else { continue };
        if a.rank() != 2 {
            continue;
        };
        if op.k_axis().inputs[slot][0] == 0 {
            let mut patch = TypedModelPatch::default();
            let mut taps = patch.taps(model, &node.inputs)?;
            taps[slot] = patch.wire_node(
                format!("{}.t_{}", &node.name, slot),
                AxisOp::Move(1, 0),
                &[taps[slot]],
            )?[0];
            let mut new_op = op.clone();
            new_op.op.axes = op
                .op
                .axes
                .clone()
                .remove_axis_occurency(InOut::In(slot), 0)?
                .with_extra_axis_occurency(op.k_axis, InOut::In(slot), 1)?;
            let output = patch.wire_node(prefix, new_op, &taps)?;
            patch.shunt_outside(model, node.id.into(), output[0])?;
            return Ok(Some(patch));
        }
        let format = Q4_0;
        let mut patch = TypedModelPatch::default();
        let weights = if a.datum_type() == f16::datum_type() {
            format.quant_f16(a.as_slice::<f16>()?)?
        } else {
            format.quant_f32(a.cast_to::<f32>()?.as_slice::<f32>()?)?
        };
        let name = &model.node(node.inputs[0].node).name;
        let fact = BlockQuantFact::new(Box::new(format), a.shape().into());
        let value = BlockQuantValue { fact: fact.clone(), value: Arc::new(weights) };
        let weights = patch.wire_node(
            format!("{name}.bq"),
            Const::new_with_opaque_fact(rctensor0(Opaque(Arc::new(value))), Box::new(fact))?,
            &[],
        )?;
        let tap = patch.tap_model(model, node.inputs[1])?;
        let wire = patch.wire_node(prefix, op.op.clone(), &[weights[0], tap])?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

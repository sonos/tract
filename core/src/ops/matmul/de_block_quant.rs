use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0};

use crate::internal::*;
use crate::ops::einsum::einsum_matmul::EinSumMatMul;
use crate::ops::konst::Const;
use crate::transform::ModelTransform;

#[derive(Debug)]
pub struct BlockQuantTransform;

impl ModelTransform for BlockQuantTransform {
    fn name(&self) -> StaticName {
        "block_quant".into()
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
    rule_if!(node.inputs.len() == 2);
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
            format.quant_f16(a.try_as_plain()?.as_slice::<f16>()?)?
        } else {
            format.quant_f32(a.cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?)?
        };
        let name = &model.node(node.inputs[0].node).name;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let bqs = BlockQuantStorage::new(Box::new(format), m, k, Arc::new(weights))?;
        let fact =
            Box::new(BlockQuantFact::new(dyn_clone::clone_box(bqs.format()), tvec!(1, m, k)));
        let weights = patch.wire_node(
            format!("{name}.bq"),
            Const::new_with_exotic_fact(
                Arc::new(bqs.into_tensor_with_shape(a.datum_type(), &[1, m, k])),
                fact,
            )?,
            &[],
        )?;
        let tap = patch.tap_model(model, node.inputs[1])?;
        // Block-quant tensor is rank 3 [G=1, M, K]; add a group dim to the axes
        let mut new_op = op.op.clone();
        new_op.axes = new_op.axes.with_extra_axis('G', InOut::In(slot), 0)?;
        let wire = patch.wire_node(prefix, new_op, &[weights[0], tap])?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

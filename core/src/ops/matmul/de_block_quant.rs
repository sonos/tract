use tract_linalg::block_quant::{BlockQuant, BlockQuantFact, BlockQuantStorage, Q4_0};

use crate::internal::*;
use crate::ops::einsum::einsum_matmul::EinSumMatMul;
use crate::ops::konst::Const;
use crate::ops::matmul::apex_policy::{self, ApexProfile};
use crate::transform::ModelTransform;

#[derive(Debug)]
pub struct BlockQuantTransform;

/// Which weights `apply_weight_quant` rewrites, and to which block-quant format.
///
/// `Rewriter` requires its context type to implement `Default`. `Uniform` is that
/// default; `apex_moe` passes `ApexBalanced` explicitly.
#[derive(Clone, Debug, Default)]
pub enum QuantMode {
    /// Every rank-2 matmul constant becomes Q4_0.
    #[default]
    Uniform,
    /// APEX MoE balanced: format depends on the weight name and layer index.
    ApexBalanced { layers: usize },
}

impl ModelTransform for BlockQuantTransform {
    fn name(&self) -> StaticName {
        "block_quant".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        apply_weight_quant(model, &QuantMode::Uniform)
    }
}

pub fn apply_weight_quant(model: &mut TypedModel, mode: &QuantMode) -> TractResult<()> {
    crate::ops::einsum::einsum_matmul::detect_all(model)?;
    Rewriter::<QuantMode>::default()
        .with_rule_for("block_quant_einsum_weights", block_quant_einsum_weights)
        .rewrite(mode, model)?;
    crate::ops::einsum::einsum_matmul::flatten_all(model)?;
    Ok(())
}

/// Name the quantizer keys off. A K-axis move is inserted before quantization, so the
/// einsum input on the second pass is that move, not the constant. Walk back to the constant.
fn weight_node_name(model: &TypedModel, mut node_id: usize) -> String {
    loop {
        let node = model.node(node_id);
        if node.op_is::<Const>() || node.inputs.len() != 1 {
            return node.name.clone();
        }
        node_id = node.inputs[0].node;
    }
}

fn format_for(mode: &QuantMode, name: &str) -> Box<dyn BlockQuant> {
    match *mode {
        QuantMode::Uniform => Box::new(Q4_0),
        QuantMode::ApexBalanced { layers } => {
            apex_policy::assign(ApexProfile::Balanced, layers, name)
        }
    }
}

fn block_quant_einsum_weights(
    mode: &QuantMode,
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
        let weight_name = weight_node_name(model, node.inputs[slot].node);
        let kpos = op.k_axis().inputs[slot][0];
        let format = format_for(mode, &weight_name);
        if a.shape()[kpos] % format.block_len() != 0 {
            continue;
        }
        if kpos == 0 {
            let mut patch = TypedModelPatch::default();
            let mut taps = patch.taps(model, &node.inputs)?;
            // Keep the constant's name on the move. Const-prop folds the move into a fresh
            // constant and drops the original, and the next pass reads that folded name.
            taps[slot] =
                patch.wire_node(format!("{weight_name}.k"), AxisOp::Move(1, 0), &[taps[slot]])?[0];
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
        let mut patch = TypedModelPatch::default();
        let weights = if a.datum_type() == f16::datum_type() {
            format.quant_f16(a.try_as_plain_ram()?.as_slice::<f16>()?)?
        } else {
            format.quant_f32(a.cast_to::<f32>()?.try_as_plain_ram()?.as_slice::<f32>()?)?
        };
        let act_slot = 1 - slot;
        let name = &weight_name;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let bqs = BlockQuantStorage::new(dyn_clone::clone_box(&*format), m, k, Arc::new(weights))?;
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
        let tap = patch.tap_model(model, node.inputs[act_slot])?;
        // Block-quant tensor is rank 3 [G=1, M, K]; add a group dim to the weight's axes
        let mut new_op = op.op.clone();
        new_op.axes = new_op.axes.with_extra_axis('G', InOut::In(slot), 0)?;
        let inputs = if slot == 0 { [weights[0], tap] } else { [tap, weights[0]] };
        let wire = patch.wire_node(prefix, new_op, &inputs)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::einsum::EinSum;

    // Deterministic varied fill so quantization is actually exercised.
    fn fill(shape: &[usize], seed: usize) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> =
            (0..n).map(|i| (((i * 13 + seed * 7) % 29) as f32 - 14.0) / 14.0).collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    fn build(axes: &str, x_shape: &[usize], w: &Tensor) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact(x_shape))?;
        let w = model.wire_node("w", Const::new(w.clone().into_arc_tensor())?, &[])?[0];
        let out = model.wire_node(
            "mm",
            EinSum { axes: axes.parse()?, operating_dt: f32::datum_type(), q_params: None },
            &[x, w],
        )?;
        model.select_output_outlets(&out)?;
        model.into_decluttered()
    }

    fn eval(model: TypedModel, x: &Tensor) -> TractResult<Tensor> {
        let out = model.into_runnable()?.run(tvec!(x.clone().into_tvalue()))?;
        Ok(out[0].clone().into_tensor())
    }

    // `X @ W` with W a rank-2 const (`w_k_axis` = contraction axis within W).
    // Asserts BlockQuantTransform both applies AND computes `X @ Q4_0(W)` correctly:
    // reference uses the same Q4_0-dequantized W, isolating operand wiring from quant error.
    fn check(axes: &str, x_shape: &[usize], w_shape: &[usize], w_k_axis: usize) -> TractResult<()> {
        let x = fill(x_shape, 1);
        let w = fill(w_shape, 2);

        // Q4_0 blocks along the last axis, so dequantize with k moved last, then back.
        let last = w.rank() - 1;
        let w_deq = Q4_0
            .simulate_precision_loss(w.clone().move_axis(w_k_axis, last)?, last)?
            .move_axis(last, w_k_axis)?;
        let reference = eval(build(axes, x_shape, &w_deq)?, &x)?;

        let mut quant = build(axes, x_shape, &w)?;
        BlockQuantTransform.transform(&mut quant)?;
        let got = eval(quant, &x)?;

        got.close_enough(&reference, Approximation::Approximate)
    }

    #[test]
    fn block_quant_xw_rank2() -> TractResult<()> {
        // plain 2D, ONNX X@W orientation: X[m,k] @ W[k,n], k is W axis 0
        check("mk,kn->mn", &[7, 256], &[256, 256], 0)
    }

    #[test]
    fn block_quant_xw_batched() -> TractResult<()> {
        // BERT-style: X[b,m,k] @ W[k,n] -> [b,m,n]
        check("bmk,kn->bmn", &[2, 7, 256], &[256, 256], 0)
    }

    #[test]
    fn block_quant_weights_already_nk() -> TractResult<()> {
        // canonical orientation (k inner on W): X[m,k] @ W[n,k], k is W axis 1
        check("mk,nk->mn", &[7, 256], &[256, 256], 1)
    }
}

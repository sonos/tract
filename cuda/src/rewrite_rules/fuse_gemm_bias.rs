use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::ops::math::Add;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::rule_ensure;

use crate::ops::CudaGgmlGemm;

/// Search pattern => GEMM(A, W) [-> axis ops] + BIAS, an f16 GEMM whose output reaches the
/// Add only through axis ops, BIAS a const of `n` elements along the axis the GEMM's last axis
/// lands on. Rewrites to GEMM(A, W, BIAS), which adds the bias in the GEMM's epilogue, followed
/// by the same axis ops.
///
/// Runs right after translation, before the axis-op fusions wrap the GEMM and the Add.
pub fn fuse_gemm_bias(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &GpuBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.mini_op.is::<Add>());
    rule_if_some!(slot = (0..2).find(|&s| model.node(node.inputs[s].node).op_is::<Const>()));
    let (bias, x) = (node.inputs[slot], node.inputs[1 - slot]);

    let mut axis_ops = tvec!();
    let mut wire = x;
    let gemm = loop {
        let prev = model.node(wire.node);
        rule_ensure!(prev.outputs.len() == 1 && prev.outputs[0].successors.len() == 1);
        let Some(axis_op) = prev.op_as::<GpuAxisOp>() else { break prev };
        axis_ops.push((prev.name.clone(), axis_op.clone()));
        wire = prev.inputs[0];
    };
    rule_ensure!(gemm.op_is::<CudaGgmlGemm>() && gemm.inputs.len() == 2);
    axis_ops.reverse();

    let fact = |o: OutletId| -> TractResult<TypedFact> {
        let f = model.outlet_fact(o)?;
        Ok(f.as_device_fact().map(|d| (**d).clone()).unwrap_or_else(|| f.clone()))
    };
    let gemm_inputs = model.node_input_facts(gemm.id)?;
    for f in gemm_inputs {
        let f = f.as_device_fact().map(|d| &**d).unwrap_or(f);
        rule_ensure!(f.datum_type == DatumType::F16 && f.is_plain());
    }
    let gemm_out = fact(gemm.id.into())?;
    let (x_fact, bias_fact, out_fact) = (fact(x)?, fact(bias)?, fact(node.id.into())?);
    rule_ensure!(bias_fact.datum_type == DatumType::F16 && out_fact.datum_type == DatumType::F16);
    rule_ensure!(out_fact.shape == x_fact.shape);

    let mut axis = gemm_out.rank() - 1;
    for (_, axis_op) in &axis_ops {
        rule_if_some!(moved = axis_op.inner.transform_axis(axis));
        axis = moved;
    }
    let n = &gemm_out.shape[gemm_out.rank() - 1];
    let offset = x_fact.rank() - bias_fact.rank().min(x_fact.rank());
    rule_ensure!(bias_fact.rank() <= x_fact.rank() && axis >= offset);
    for (ix, d) in bias_fact.shape.iter().enumerate() {
        rule_ensure!(if ix + offset == axis { d == n } else { d.is_one() });
    }

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &[gemm.inputs[0], gemm.inputs[1], bias])?;
    let mut wire = patch.wire_node(format!("{}.bias", gemm.name), CudaGgmlGemm, &inputs)?;
    for (name, axis_op) in axis_ops {
        wire = patch.wire_node(format!("{name}.bias"), axis_op, &wire)?;
    }
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

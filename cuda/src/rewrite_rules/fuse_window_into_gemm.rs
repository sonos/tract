use tract_core::internal::*;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::pulse::GpuDelay;
use tract_gpu::rule_ensure;

use crate::ops::{CudaGgmlGemm, CudaRingGemm};

/// A pulsed window writing the operand its GEMM reads, and nothing else,
/// becomes [`CudaRingGemm`]: the window stays a ring the kernel rotates as it
/// reads, so the turn writes one slot instead of the whole window.
///
/// Runs between the output-side and input-side axis-op fusions: the first is
/// what puts the window in the GEMM's layout, the second would wrap both nodes
/// and hide them.
pub fn fuse_window_into_gemm(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    delay: &GpuDelay,
) -> TractResult<Option<TypedModelPatch>> {
    let op = &delay.inner;
    // The window is all of the ring, so the ring can hold no delay to wait out,
    // and its zeroed slots have to be what the stream's start reads.
    rule_ensure!(op.delay == 0 && op.zero_pad && op.overlap > 0);
    rule_ensure!(!delay.out_axis_ops.is_empty());

    let slot = model.node_input_facts(node.id)?[0];
    let slot_shape = slot.as_device_fact().map(|f| &f.shape).unwrap_or(&slot.shape);
    rule_ensure!(slot_shape[op.axis].is_one());

    rule_if_some!(gemm = model.single_succ(node.id)?);
    rule_ensure!(gemm.op_is::<CudaGgmlGemm>());
    rule_ensure!(gemm.inputs[1] == node.id.into());

    let window = model.node_output_facts(node.id)?[0].clone();
    let act = model.node_input_facts(gemm.id)?[0];
    let (window, act) = (
        window.as_device_fact().map(|f| f.clone().into_typed_fact()).unwrap_or(window),
        act.as_device_fact().map(|f| f.clone().into_typed_fact()).unwrap_or(act.clone()),
    );
    // The kernel reading a ring is the matvec one, and the window axis has to be
    // its rows: whole slots of them, so a slot rotation is a row rotation.
    rule_ensure!(act.rank() >= 2 && window.rank() == act.rank());
    let out = CudaGgmlGemm.resolve_output_facts(&[&act, &window])?;
    rule_if_some!(m = out[0].shape[out[0].rank() - 2].as_i64());
    rule_if_some!(k = act.shape[act.rank() - 1].as_i64());
    rule_if_some!(n = window.shape[window.rank() - 2].as_i64());
    rule_ensure!(m <= 8 && k % 2 == 0);
    rule_ensure!((n as usize).is_multiple_of(op.overlap + 1));

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &[gemm.inputs[0], node.inputs[0]])?;
    let wire = patch.wire_node(
        gemm.name.clone(),
        CudaRingGemm { delay: op.clone(), window_axis_ops: delay.out_axis_ops.clone() },
        &inputs,
    )?;
    patch.shunt_outside(model, gemm.id.into(), wire[0])?;
    Ok(Some(patch))
}

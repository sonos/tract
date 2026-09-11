use tract_core::internal::*;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::pulse::GpuDelay;
use tract_gpu::rule_ensure;

use crate::ops::{CudaGgmlGemm, CudaRingGemm};

/// A pulsed window writing the operand its GEMM reads, and nothing else,
/// becomes [`CudaRingGemm`]: the window stays a ring the kernel rotates as it
/// reads, so the turn writes one slot instead of the whole window. The ring
/// turns on the window's rows or on the axis the GEMM contracts, whichever the
/// layout between the window and its GEMM puts the window axis in.
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

    let slot = model.node_input_facts(node.id)?[0];
    let slot_shape = slot.as_device_fact().map(|f| &f.shape).unwrap_or(&slot.shape);
    rule_ensure!(slot_shape[op.axis].is_one());

    // The window reaches its GEMM through the chain the delay writes through
    // and any axis op still standing between the two, each of which the ring
    // stands for by being held in the layout they lead to.
    let mut window_axis_ops = delay.out_axis_ops.clone();
    let mut window_wire = node.id;
    let gemm = loop {
        rule_if_some!(succ = model.single_succ(window_wire)?);
        let Some(axis_op) = succ.op_as::<GpuAxisOp>() else { break succ };
        window_axis_ops.push(axis_op.clone());
        window_wire = succ.id;
    };
    rule_ensure!(!window_axis_ops.is_empty());
    rule_ensure!(gemm.op_is::<CudaGgmlGemm>());
    rule_ensure!(gemm.inputs[1] == window_wire.into());

    let window = model.node_input_facts(gemm.id)?[1].clone();
    let act = model.node_input_facts(gemm.id)?[0];
    let (window, act) = (
        window.as_device_fact().map(|f| f.clone().into_typed_fact()).unwrap_or(window),
        act.as_device_fact().map(|f| f.clone().into_typed_fact()).unwrap_or(act.clone()),
    );
    // The kernel reading a ring is the matvec one, and the axis it turns has to
    // hold whole slots: rows of them, or an even run of columns of every row,
    // the kernel reading the contracted axis two columns at a time.
    rule_ensure!(act.rank() >= 2 && window.rank() == act.rank());
    let out = CudaGgmlGemm.resolve_output_facts(&[&act, &window])?;
    rule_if_some!(m = out[0].shape[out[0].rank() - 2].as_i64());
    rule_if_some!(k = act.shape[act.rank() - 1].as_i64());
    rule_if_some!(n = window.shape[window.rank() - 2].as_i64());
    rule_ensure!(m <= 8 && k % 2 == 0);
    let slots = op.overlap + 1;
    let (n, k) = (n as usize, k as usize);
    rule_ensure!(
        n.is_multiple_of(slots) || (k.is_multiple_of(slots) && (k / slots).is_multiple_of(2))
    );

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &[gemm.inputs[0], node.inputs[0]])?;
    let wire = patch.wire_node(
        gemm.name.clone(),
        CudaRingGemm { delay: op.clone(), window_axis_ops },
        &inputs,
    )?;
    patch.shunt_outside(model, gemm.id.into(), wire[0])?;
    Ok(Some(patch))
}

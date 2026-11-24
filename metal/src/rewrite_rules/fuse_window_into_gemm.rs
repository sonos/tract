use tract_core::internal::*;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::pulse::GpuDelay;
use tract_gpu::rule_ensure;

use crate::kernels::matmul::ring::RING_MATVEC_MAX_M;
use crate::kernels::matmul::{BasicMatMul, GgmlGemm, MfaGemm, MlxGemm, MppGemm};
use crate::ops::{MetalGemm, MetalRingGemm};

/// `transpose_b` of a Metal GEMM of `a [.., m, k]` (not transposed) by `b`.
fn untransposed_a_gemm(node: &TypedNode) -> Option<bool> {
    fn of<K: crate::kernels::matmul::GemmKernel + 'static>(node: &TypedNode) -> Option<bool> {
        node.op_as::<MetalGemm<K>>().filter(|g| !g.kernel.transpose_a).map(|g| g.kernel.transpose_b)
    }
    of::<MppGemm>(node)
        .or_else(|| of::<MlxGemm>(node))
        .or_else(|| of::<GgmlGemm>(node))
        .or_else(|| of::<MfaGemm>(node))
        .or_else(|| of::<BasicMatMul>(node))
}

/// A pulsed window feeding a GEMM's second operand, and nothing else, becomes
/// [`MetalRingGemm`]: the window stays a ring the kernel rotates as it reads,
/// so a turn writes one slot instead of the whole window. The ring turns on
/// the window's contracted axis or on its output columns, whichever the axis
/// ops between the window and its GEMM put the window axis in.
///
/// Runs before the axis-op fusion, which would wrap both nodes and hide the
/// chain between them.
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

    let mut window_axis_ops = delay.out_axis_ops.clone();
    let mut window_wire = node.id;
    let gemm = loop {
        rule_if_some!(succ = model.single_succ(window_wire)?);
        let Some(axis_op) = succ.op_as::<GpuAxisOp>() else { break succ };
        window_axis_ops.push(axis_op.clone());
        window_wire = succ.id;
    };
    rule_if_some!(transpose_b = untransposed_a_gemm(gemm));
    rule_ensure!(gemm.inputs[1] == window_wire.into());

    let facts = model.node_input_facts(gemm.id)?;
    let (act, window) = (
        facts[0].as_device_fact().map(|f| f.clone().into_typed_fact()).unwrap_or(facts[0].clone()),
        facts[1].as_device_fact().map(|f| f.clone().into_typed_fact()).unwrap_or(facts[1].clone()),
    );
    rule_ensure!(act.is_plain() && window.is_plain());
    rule_ensure!(matches!(act.datum_type, DatumType::F16 | DatumType::F32));
    rule_ensure!(act.datum_type == window.datum_type);
    let rank = act.rank();
    rule_ensure!(rank >= 2 && window.rank() == rank);
    rule_ensure!(act.shape[..rank - 2] == window.shape[..rank - 2]);
    rule_if_some!(m = act.shape[rank - 2].as_i64());
    rule_if_some!(rows = window.shape[rank - 2].as_i64());
    rule_if_some!(cols = window.shape[rank - 1].as_i64());
    rule_ensure!(m as usize <= RING_MATVEC_MAX_M);
    let slots = op.overlap + 1;
    rule_ensure!((rows as usize).is_multiple_of(slots) || (cols as usize).is_multiple_of(slots));

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &[gemm.inputs[0], node.inputs[0]])?;
    let wire = patch.wire_node(
        gemm.name.clone(),
        MetalRingGemm { delay: op.clone(), window_axis_ops, transpose_b },
        &inputs,
    )?;
    patch.shunt_outside(model, gemm.id.into(), wire[0])?;
    Ok(Some(patch))
}

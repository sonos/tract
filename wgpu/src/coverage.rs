//! Load-time coverage check.
//!
//! Without JSPI a mid-graph GPU→CPU readback cannot complete in a browser.
//! [`ensure_wgpu_coverage`] still fails with named ops for that case.
//! The wgpu `Runtime::prepare` path skips it when
//! [`crate::hybrid_fallback_available`] (native blocking poll, or wasm built
//! with `--features jspi` on a JSPI browser): uncovered ops run on tract's
//! CPU kernels instead.

use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::ops::source::TypedSource;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::sync::{DeviceSync, DeviceSyncKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UncoveredOp {
    pub node: String,
    pub op: String,
}

fn is_glue(node: &TypedNode) -> bool {
    node.op_is::<TypedSource>() || node.op_is::<Const>() || node.op_is::<DeviceSync>()
}

/// Nodes the backend cannot run, in eval order. Glue (Source, Const, DeviceSync)
/// is ignored; everything else must produce `DeviceFact` outputs.
pub fn uncovered_ops(model: &TypedModel) -> TractResult<Vec<UncoveredOp>> {
    let mut out = Vec::new();
    for id in model.eval_order()? {
        let node = &model.nodes()[id];
        if is_glue(node) {
            continue;
        }
        let facts = model.node_output_facts(id)?;
        if facts.iter().all(|f| f.as_device_fact().is_some()) {
            continue;
        }
        out.push(UncoveredOp { node: node.name.to_string(), op: node.op.name().to_string() });
    }
    Ok(out)
}

/// Nodes that read a tensor brought back from the device, in eval order: each needs a
/// readback before the model is done. Nodes on the CPU fed from the host alone, such as a
/// prologue over the model's inputs, are not among them.
pub fn mid_graph_readbacks(model: &TypedModel) -> TractResult<Vec<UncoveredOp>> {
    let mut out = Vec::new();
    for id in model.eval_order()? {
        let node = &model.nodes()[id];
        if !node.op_as::<DeviceSync>().is_some_and(|s| s.kind == DeviceSyncKind::ToHost) {
            continue;
        }
        for succ in &node.outputs[0].successors {
            let reader = model.node(succ.node);
            out.push(UncoveredOp {
                node: reader.name.to_string(),
                op: reader.op.name().to_string(),
            });
        }
    }
    Ok(out)
}

/// Reject a model that would read back from the device before it is done.
pub fn ensure_wgpu_coverage(model: &TypedModel) -> TractResult<()> {
    let bad = mid_graph_readbacks(model)?;
    if bad.is_empty() {
        return Ok(());
    }
    let list =
        bad.iter().map(|u| format!("{} (node {:?})", u.op, u.node)).collect::<Vec<_>>().join(", ");
    bail!(
        "tract-wgpu cannot run this model; no GPU kernel for: {list}. \
         Mid-graph CPU fallback is not supported without JSPI."
    )
}

use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::nn::Sigmoid;
use tract_gpu::rule_ensure;
use tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent;

/// Fold a singleton `Sigmoid` feeding the beta input of a
/// `GatedDeltaNetRecurrent` into the op itself (`sigmoid_beta`): the Metal
/// kernel replicates the elementwise sigmoid on half bit-exactly, so this
/// only removes the standalone dispatch. The sigmoid node must have no other
/// consumer (beta is per-(step, head), nothing else reads it in practice;
/// a shared output would keep the sigmoid alive AND fold, still correct but
/// no dispatch win, so we skip it).
pub fn fold_gdn_beta_sigmoid(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &GatedDeltaNetRecurrent,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(!op.sigmoid_beta);
    rule_ensure!(std::env::var_os("TRACT_METAL_DISABLE_GDN_BETA_SIGMOID").is_none());
    let beta_outlet = node.inputs[4];
    let beta_node = model.node(beta_outlet.node);
    let Some(ew) = beta_node.op_as::<ElementWiseOp>() else { return Ok(None) };
    rule_ensure!(ew.0.is::<Sigmoid>());
    rule_ensure!(beta_node.outputs[beta_outlet.slot].successors.len() == 1);
    rule_ensure!(beta_node.inputs.len() == 1);

    let mut patch = TypedModelPatch::new(format!("fold sigmoid into {node_name} beta"));
    let mut inputs = tvec![];
    for (ix, input) in node.inputs.iter().enumerate() {
        let outlet = if ix == 4 { beta_node.inputs[0] } else { *input };
        inputs.push(patch.tap_model(model, outlet)?);
    }
    let out = patch.wire_node(
        node_name,
        GatedDeltaNetRecurrent { sigmoid_beta: true },
        &inputs,
    )?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    patch.shunt_outside(model, OutletId::new(node.id, 1), out[1])?;
    Ok(Some(patch))
}

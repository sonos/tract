use tract_core::internal::*;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::rule_ensure;

use crate::kernels::shaders::{BINARY_OPS, ChainStep, ELEMENT_WISE_OPS};
use crate::ops::chain::WgpuElementWiseChain;
use crate::ops::conv::WgpuConv;
use crate::ops::deconv::WgpuDeconv;
use crate::ops::matmul::WgpuGemm;

/// Epilogue operands ride in one `vec4` of offsets.
const MAX_EPILOGUE_OPERANDS: usize = 3;

fn shape_of(model: &TypedModel, outlet: OutletId) -> TractResult<TVec<TDim>> {
    let fact = model.outlet_fact(outlet)?;
    Ok(fact
        .as_device_fact()
        .map(|f| f.shape.dims().into())
        .unwrap_or_else(|| fact.shape.dims().into()))
}

/// A GEMM's epilogue can only read an operand that is one value, or one per
/// output column — that is what a convolution bias becomes here.
fn addressable(model: &TypedModel, outlet: OutletId, columns: &TDim) -> TractResult<bool> {
    let shape = shape_of(model, outlet)?;
    let len = shape.iter().product::<TDim>();
    Ok(len == 1.to_dim() || len == *columns)
}

fn steps_of(node: &TypedNode, slot: usize, rhs: usize) -> Option<(Vec<ChainStep>, usize)> {
    if let Some(chain) = node.op_as::<WgpuElementWiseChain>() {
        return Some((chain.steps.clone(), node.inputs.len() - 1));
    }
    if let Some(op) = node.op_as::<GpuElementWise>() {
        let name = op.mini_op.name().to_lowercase();
        return ELEMENT_WISE_OPS
            .contains(&name.as_str())
            .then(|| (vec![ChainStep::Unary(name)], 0));
    }
    let op = node.op_as::<GpuBinOp>()?;
    let name = op.mini_op.name().to_lowercase();
    BINARY_OPS
        .contains(&name.as_str())
        .then(|| (vec![ChainStep::Binary { op: name, rhs, swapped: slot == 1 }], 1))
}

/// Folds the elementwise ops that follow a convolution into it, the same way.
/// Folds the elementwise ops that follow a transposed convolution into it,
/// the same way: on a segmenter that is the mask's bias and sigmoid.
pub fn fuse_deconv_epilogue(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &WgpuDeconv,
) -> TractResult<Option<TypedModelPatch>> {
    let channels = op.op.pool_spec.output_channels.to_dim();
    absorb(model, node, node_name, channels, &op.epilogue, |steps| WgpuDeconv {
        op: op.op.clone(),
        epilogue: steps,
    })
}

pub fn fuse_conv_epilogue(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &WgpuConv,
) -> TractResult<Option<TypedModelPatch>> {
    let channels = op.op.pool_spec.output_channels.to_dim();
    absorb(model, node, node_name, channels, &op.epilogue, |steps| WgpuConv {
        op: op.op.clone(),
        epilogue: steps,
    })
}

/// Folds the elementwise ops that follow a GEMM into the GEMM itself, so a
/// bias and its activation cost no extra pass over the result.
pub fn fuse_gemm_epilogue(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &WgpuGemm,
) -> TractResult<Option<TypedModelPatch>> {
    let out_shape = shape_of(model, node.id.into())?;
    rule_ensure!(out_shape.len() >= 2);
    let columns = if op.op.transpose_c {
        out_shape[out_shape.len() - 2].clone()
    } else {
        out_shape[out_shape.len() - 1].clone()
    };
    absorb(model, node, node_name, columns, &op.epilogue, |steps| WgpuGemm {
        op: op.op,
        epilogue: steps,
    })
}

/// An axis op that leaves every value where it is, so an epilogue can be
/// applied before it as well as after.
fn is_respelling(node: &TypedNode) -> bool {
    node.op_as::<GpuAxisOp>()
        .is_some_and(|a| matches!(a.inner, AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Reshape(..)))
}

/// Grows the epilogue `existing` by the elementwise op that consumes the
/// result, stepping over the respellings in between; those are re-wired after
/// the fused op so its output keeps its own shape.
fn absorb<O: TypedOp>(
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    columns: TDim,
    existing: &[ChainStep],
    build: impl FnOnce(Vec<ChainStep>) -> O,
) -> TractResult<Option<TypedModelPatch>> {
    let mut respellings: Vec<&TypedNode> = vec![];
    let mut tail = node;
    let succ = loop {
        let Some(succ) = model.single_succ(tail.id)? else { return Ok(None) };
        if is_respelling(succ) {
            respellings.push(succ);
            tail = succ;
        } else {
            break succ;
        }
    };
    rule_ensure!(succ.inputs.iter().filter(|i| i.node == tail.id).count() == 1);
    rule_ensure!(shape_of(model, succ.id.into())? == shape_of(model, tail.id.into())?);

    let existing_extras = node.inputs.len() - 2;
    let slot = succ.inputs.iter().position(|i| i.node == tail.id).unwrap();
    let Some((steps, operands)) = steps_of(succ, slot, 1) else { return Ok(None) };
    rule_ensure!(existing_extras + operands <= MAX_EPILOGUE_OPERANDS);
    let steps = steps.into_iter().map(|s| match s {
        ChainStep::Binary { op, rhs, swapped } => {
            ChainStep::Binary { op, rhs: rhs + existing_extras, swapped }
        }
        other => other,
    });

    let extras: TVec<OutletId> = succ
        .inputs
        .iter()
        .enumerate()
        .filter(|(i, inlet)| {
            !(succ.op_is::<WgpuElementWiseChain>() && *i == 0) && inlet.node != tail.id
        })
        .map(|(_, inlet)| *inlet)
        .collect();
    rule_ensure!(extras.len() == operands);
    for extra in &extras {
        rule_ensure!(addressable(model, *extra, &columns)?);
    }

    let mut patch = TypedModelPatch::default();
    let mut inputs = vec![];
    for input in node.inputs.iter().chain(extras.iter()) {
        inputs.push(patch.tap_model(model, *input)?);
    }
    let all_steps = existing.iter().cloned().chain(steps).collect();
    let mut out = patch.wire_node(format!("{node_name}.epilogue"), build(all_steps), &inputs)?[0];
    for respelling in &respellings {
        out = patch.wire_node(
            format!("{}.epilogue", respelling.name),
            respelling.op.clone(),
            &[out],
        )?[0];
    }
    patch.shunt_outside(model, succ.id.into(), out)?;
    Ok(Some(patch))
}

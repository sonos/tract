use tract_core::internal::*;
use tract_core::ops::math::Sub;
use tract_core::ops::nn::Sigmoid;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::cast::GpuCast;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::concat::GpuConcat;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::ops::slice::GpuSlice;
use tract_gpu::rule_ensure;

use crate::kernels::bin_ops::BINARY_OPS;
use crate::kernels::conv::DECODER_STATE_UPDATE;
use crate::kernels::conv::decoder_state_epilogue;
use crate::kernels::element_wise::ELEMENT_WISE_OPS;
use crate::kernels::shaders::ChainStep;
use crate::ops::chain::WgpuElementWiseChain;
use crate::ops::conv::WgpuConv;
use crate::ops::deconv::WgpuDeconv;
use crate::ops::matmul::WgpuGemm;

/// Fuses a gated FP16 recurrent update into a direct NCHW convolution.
pub fn fuse_decoder_state_update(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &WgpuConv,
) -> TractResult<Option<TypedModelPatch>> {
    let context = crate::wgpu_context();
    rule_ensure!(
        context.apple_hardware_webgpu()
            && context.shader_f16()
            && context.device().limits().max_storage_buffers_per_shader_stage >= 7
    );
    let spec = &conv.op.pool_spec;
    rule_ensure!(
        conv.epilogue.is_empty()
            && conv.f16_accumulation
            && conv.op.group == 1
            && spec.data_format == tract_core::ops::nn::DataFormat::NCHW
            && spec.kernel_shape[..] == [3, 3]
            && spec.strides()[..] == [1, 1]
            && spec.dilations()[..] == [1, 1]
    );
    let input = model.outlet_fact(node.inputs[0])?.to_device_fact()?;
    let output = model.outlet_fact(node.id.into())?.to_device_fact()?;
    rule_ensure!(input.datum_type == DatumType::F16 && output.datum_type == DatumType::F16);
    let Some([1, c, h, w]) = output.shape.as_concrete() else { return Ok(None) };
    rule_ensure!(matches!(
        (*c, *h, *w),
        (64, 23, 40) | (40, 45, 80) | (20, 90, 160) | (16, 180, 320)
    ));
    rule_ensure!(input.shape.as_concrete() == Some(&[1, 2 * c, *h, *w]));
    let pad = spec.computed_padding(&input.shape.dims()[2..]);
    rule_ensure!(pad.iter().all(|p| p.pad_before == 1.to_dim() && p.pad_after == 1.to_dim()));
    let Some(concat) = model.node(node.inputs[0].node).op_as::<GpuConcat>() else {
        return Ok(None);
    };
    rule_ensure!(concat.axis == 1);
    let Some(term) = model.single_succ(node.id)? else { return Ok(None) };
    let Some(activation) = term.op_as::<WgpuElementWiseChain>() else { return Ok(None) };
    let [head, bias, gate] = term.inputs.as_slice() else { return Ok(None) };
    rule_ensure!(*head == OutletId::from(node.id));
    let bias_fact = model.outlet_fact(*bias)?.to_device_fact()?;
    rule_ensure!(
        bias_fact.datum_type == DatumType::F16
            && bias_fact.shape.as_concrete().is_some_and(|shape| {
                shape.iter().all(|&dim| dim == 1) || shape == [1, *c, 1, 1]
            })
    );
    let binary = |op: &str, rhs, swapped| ChainStep::Binary { op: op.into(), rhs, swapped };
    rule_ensure!(
        activation.steps
            == [binary("add", 1, false), ChainStep::Unary("tanh".into()), binary("mul", 2, true)]
    );
    let Some(state) = model.single_succ(term.id)? else { return Ok(None) };
    let Some(blend) = state.op_as::<WgpuElementWiseChain>() else { return Ok(None) };
    let [inverse, prev, term_in] = state.inputs.as_slice() else { return Ok(None) };
    rule_ensure!(
        *term_in == OutletId::from(term.id)
            && blend.steps == [binary("mul", 1, false), binary("add", 2, false)]
    );
    let subtract = model.node(inverse.node);
    let [one, gate_in] = subtract.inputs.as_slice() else { return Ok(None) };
    rule_ensure!(
        *gate_in == *gate
            && subtract
                .op_as::<GpuBinOp>()
                .is_some_and(|op| op.backend_name == "Wgpu" && op.mini_op.is::<Sub>())
    );
    let start = model.outlet_fact(*one)?.to_device_fact()?;
    rule_ensure!(start.datum_type == DatumType::F16 && start.shape.volume() == 1.to_dim());
    let outs = model.output_outlets()?;
    rule_ensure!(!outs.contains(head) && !outs.contains(term_in));
    for o in [term.id.into(), state.id.into(), *inverse, *gate, *prev] {
        let f = model.outlet_fact(o)?.to_device_fact()?;
        rule_ensure!(f.datum_type == DatumType::F16 && f.shape == output.shape);
    }
    let gate_node = model.node(gate.node);
    rule_ensure!(
        gate_node.op_is::<GpuSlice>()
            || gate_node
                .op_as::<GpuElementWise>()
                .is_some_and(|op| { op.backend_name == "Wgpu" && op.mini_op.is::<Sigmoid>() })
    );
    let previous = model.node(prev.node);
    let [previous_input] = previous.inputs.as_slice() else { return Ok(None) };
    rule_ensure!(
        previous
            .op_as::<GpuCast>()
            .is_some_and(|op| { op.backend_name == "Wgpu" && op.to == DatumType::F16 })
            && model.outlet_fact(*previous_input)?.datum_type == DatumType::F32
    );
    let mut patch = TypedModelPatch::default();
    let mut inputs = tvec![];
    for o in [node.inputs[0], node.inputs[1], *bias, *gate, *prev, *one] {
        inputs.push(patch.tap_model(model, o)?);
    }
    let result = patch.wire_node(
        format!("{name}.decoder-state"),
        WgpuConv { epilogue: vec![ChainStep::Unary(DECODER_STATE_UPDATE.into())], ..conv.clone() },
        &inputs,
    )?[0];
    patch.shunt_outside(model, state.id.into(), result)?;
    Ok(Some(patch))
}

/// Epilogue operands ride in one `vec4` of offsets.
const MAX_EPILOGUE_OPERANDS: usize = 3;

fn shape_of(model: &TypedModel, outlet: OutletId) -> TractResult<TVec<TDim>> {
    let fact = model.outlet_fact(outlet)?;
    Ok(fact
        .as_device_fact()
        .map(|f| f.shape.dims().into())
        .unwrap_or_else(|| fact.shape.dims().into()))
}

/// An epilogue reads an operand that is one value, or one per output column —
/// that is what a convolution bias becomes here — or, for a GEMM, one laid out
/// like the output itself, which a residual is. That last one is read through
/// the output's own strides, so it cannot come from an axis op that a later
/// pass would fold into a strided view.
fn addressable(
    model: &TypedModel,
    outlet: OutletId,
    columns: &TDim,
    output: Option<&TVec<TDim>>,
) -> TractResult<bool> {
    let shape = shape_of(model, outlet)?;
    let len = shape.iter().product::<TDim>();
    if len == 1.to_dim() || len == *columns {
        return Ok(true);
    }
    Ok(output.is_some_and(|out| *out == shape)
        && !model.node(outlet.node).op_is::<GpuAxisOp>()
        && model.node(outlet.node).op_as::<tract_core::ops::konst::Const>().is_none())
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
    absorb(model, node, node_name, channels, &op.epilogue, false, |steps| WgpuDeconv {
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
    rule_ensure!(!decoder_state_epilogue(&op.epilogue));
    let channels = op.op.pool_spec.output_channels.to_dim();
    absorb(model, node, node_name, channels, &op.epilogue, false, |steps| WgpuConv {
        op: op.op.clone(),
        epilogue: steps,
        f16_accumulation: op.f16_accumulation,
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
    absorb(model, node, node_name, columns, &op.epilogue, true, |steps| WgpuGemm {
        op: op.op,
        epilogue: steps,
        f16_accumulation: op.f16_accumulation,
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
    elementwise: bool,
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
    let out_shape = shape_of(model, node.id.into())?;
    for extra in &extras {
        rule_ensure!(addressable(model, *extra, &columns, elementwise.then_some(&out_shape))?);
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

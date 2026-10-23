use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::change_axes::AxisChange;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::ops::identity::Identity;
use tract_core::ops::nn::DataFormat;
use tract_core::optim::change_axes::change_axes;

/// Run a 2D NCHW conv as an NHWC conv between two transposes.
///
/// GPU conv kernels (cuDNN tensor cores, MLX implicit GEMM) work in NHWC. The filter is
/// reordered to OHWI here, once, as a constant, and the input and output are transposed
/// instead. Requires a constant OIHW filter, a single group and symmetric padding. Backends
/// add their own gates (datum type, shape) before calling it.
pub fn conv_to_nhwc(
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(conv.pool_spec.data_format == DataFormat::NCHW);
    rule_ensure!(conv.kernel_fmt == KernelFormat::OIHW);
    rule_ensure!(conv.group == 1 && conv.q_params.is_none());
    let input_fact = model.outlet_fact(node.inputs[0])?;
    rule_ensure!(input_fact.rank() == 4);
    rule_if_some!(kernel = &model.outlet_fact(node.inputs[1])?.konst);
    let data_shape = conv.pool_spec.data_format.shape(input_fact.shape.to_tvec())?;
    rule_ensure!(
        conv.pool_spec
            .computed_padding(data_shape.hw_dims())
            .iter()
            .all(|p| p.pad_before == p.pad_after)
    );

    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &node.inputs)?;
    inputs[0] = patch.wire_node(format!("{name}.to_nhwc"), AxisOp::Move(1, 3), &[inputs[0]])?[0];
    let ohwi = (**kernel).clone().move_axis(1, 3)?;
    inputs[1] = patch.add_const(format!("{name}.kernel_ohwi"), ohwi)?;
    let mut nhwc = conv.clone();
    nhwc.pool_spec.data_format = DataFormat::NHWC;
    nhwc.kernel_fmt = KernelFormat::OHWI;
    let wire = patch.wire_node(format!("{name}.nhwc"), nhwc, &inputs)?;
    let wire = patch.wire_node(name, AxisOp::Move(3, 1), &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

fn is_nhwc_conv(node: &TypedNode) -> bool {
    node.op_as::<Conv>().is_some_and(|c| c.pool_spec.data_format == DataFormat::NHWC)
}

/// Pushes the transposes `conv_to_nhwc` wraps around NHWC convs out into the graph, so that
/// the output transpose of one conv cancels the input transpose of the next. Run after
/// `conv_to_nhwc`, and not followed by a declutter, whose `ChangeAxes` would fold the
/// transposes back into the conv formats.
///
/// Each boundary `Move` is propagated with core's `change_axes`, with the NHWC convs' own
/// input and output wires locked, as well as the model interfaces. A region that reaches a
/// locked wire, or an op that cannot carry the change, keeps its transposes. The `Identity`
/// nodes and adjacent axis ops the propagation leaves behind are decluttered locally.
pub fn propagate_nhwc(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default().with_rule_for("propagate_nhwc", propagate_nhwc_move).rewrite(&(), model)?;
    Rewriter::default()
        .with_rule_for("declutter_identity", declutter_identity)
        .with_rule_for("declutter_axis_op", declutter_axis_op)
        .rewrite(&(), model)
}

fn propagate_nhwc_move(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &AxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!(rank = node.outputs[0].fact.rank().checked_sub(1));
    let change = match op {
        AxisOp::Move(1, to) if *to == rank => {
            let succs = &node.outputs[0].successors;
            rule_ensure!(succs.len() == 1 && succs[0].slot == 0);
            rule_ensure!(is_nhwc_conv(model.node(succs[0].node)));
            AxisChange { outlet: node.inputs[0], op: op.clone() }
        }
        AxisOp::Move(from, 1) if *from == rank => {
            rule_ensure!(is_nhwc_conv(model.node(node.inputs[0].node)));
            AxisChange { outlet: node.id.into(), op: op.recip() }
        }
        _ => return Ok(None),
    };
    let mut locked: TVec<OutletId> = model.input_outlets()?.into();
    locked.extend(model.output_outlets()?.iter().copied());
    for conv in model.nodes().iter().filter(|n| is_nhwc_conv(n)) {
        locked.push(conv.inputs[0]);
        locked.push(conv.id.into());
    }
    let mut explored = Default::default();
    Ok(change_axes(model, &change, &locked, &[], &mut explored)?.map(|(patch, _)| patch))
}

fn declutter_identity(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &Identity,
) -> TractResult<Option<TypedModelPatch>> {
    op.declutter(model, node)
}

fn declutter_axis_op(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &AxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    op.declutter(model, node)
}

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

/// Moves the channel axis of channel-last convs back in front, together with the region each one
/// heads.
///
/// Core's declutter moves a depthwise conv's channel axis last when the CPU has a channel-last
/// depthwise kernel. A backend whose kernels are channel-first calls this before lowering. Only
/// the model interfaces are locked, so a region the declutter flipped in a channel-first model
/// flips back whole, and a model that is channel-last at its interfaces keeps its layout.
pub fn restore_channels_first(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for("restore_channels_first", restore_channels_first_conv)
        .rewrite(&(), model)
}

fn restore_channels_first_conv(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    let format = conv.pool_spec.data_format;
    rule_ensure!(format.c_is_last());
    let rank = model.outlet_fact(node.inputs[0])?.rank();
    let change =
        AxisChange { outlet: node.inputs[0], op: AxisOp::Move(rank - 1, format.has_n() as usize) };
    let mut locked: TVec<OutletId> = model.input_outlets()?.into();
    locked.extend(model.output_outlets()?.iter().copied());
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

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::ops::cnn::{PaddingSpec, PoolSpec};

    fn conv(format: DataFormat, kernel: usize, ci: usize, co: usize, group: usize) -> Conv {
        let pad = kernel / 2;
        Conv {
            pool_spec: PoolSpec {
                data_format: format,
                kernel_shape: tvec![kernel, kernel],
                padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, pad]),
                dilations: None,
                strides: None,
                input_channels: ci,
                output_channels: co,
            },
            kernel_fmt: KernelFormat::OIHW,
            group,
            q_params: None,
        }
    }

    fn wire_conv(
        model: &mut TypedModel,
        name: &str,
        op: Conv,
        input: OutletId,
    ) -> TractResult<OutletId> {
        let (co, ci, k) =
            (op.output_channels(), op.input_channels() / op.group, op.pool_spec.kernel_shape[0]);
        let kernel = Tensor::from_shape(
            &[co, ci, k, k],
            &(0..co * ci * k * k).map(|i| (i % 7) as f32 * 0.1).collect::<Vec<_>>(),
        )?;
        let kernel = model.add_const(format!("{name}.k"), kernel)?;
        let bias = model.add_const(format!("{name}.b"), Tensor::zero::<f32>(&[co])?)?;
        Ok(model.wire_node(name, op, &[input, kernel, bias])?[0])
    }

    fn conv_formats(model: &TypedModel) -> Vec<DataFormat> {
        model
            .nodes()
            .iter()
            .filter_map(|n| n.op_as::<Conv>())
            .map(|c| c.pool_spec.data_format)
            .collect()
    }

    /// Pointwise, depthwise, pointwise on a channel-first input, decluttered by core. Where the
    /// CPU has a channel-last depthwise kernel, the declutter moves the depthwise channel axis
    /// last and the pointwise einsums absorb the move.
    fn decluttered_block(c: usize) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let mut wire = model.add_source("input", f32::fact([1, c, 8, 8]))?;
        wire = wire_conv(&mut model, "pw0", conv(DataFormat::NCHW, 1, c, c, 1), wire)?;
        wire = wire_conv(&mut model, "dw", conv(DataFormat::NCHW, 3, c, c, c), wire)?;
        wire = wire_conv(&mut model, "pw1", conv(DataFormat::NCHW, 1, c, c, 1), wire)?;
        model.select_output_outlets(&[wire])?;
        model.into_decluttered()
    }

    #[test]
    fn restores_a_region_core_flipped() -> TractResult<()> {
        let flipped = decluttered_block(16)?;
        if tract_linalg::routines::depthwise_c_f32().is_some() {
            ensure!(
                conv_formats(&flipped).iter().any(|f| f.c_is_last()),
                "declutter kept the layout"
            );
        }
        let mut model = flipped.clone();
        restore_channels_first(&mut model)?;
        ensure!(conv_formats(&model).iter().all(|f| !f.c_is_last()), "{:?}", conv_formats(&model));
        ensure!(model.nodes().len() == flipped.nodes().len(), "restore changed the node count");
        let input: Tensor = Tensor::from_shape(
            &[1, 16, 8, 8],
            &(0..1024).map(|i| (i % 13) as f32 - 6.).collect::<Vec<_>>(),
        )?;
        let expected = flipped.into_runnable()?.run(tvec![input.clone().into()])?;
        let got = model.into_runnable()?.run(tvec![input.into()])?;
        expected[0].close_enough(&got[0], true)
    }

    #[test]
    fn keeps_a_channel_last_model() -> TractResult<()> {
        let c = 16;
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact([1, 8, 8, c]))?;
        let out = wire_conv(&mut model, "dw", conv(DataFormat::NHWC, 3, c, c, c), input)?;
        model.select_output_outlets(&[out])?;
        let before = model.clone();
        restore_channels_first(&mut model)?;
        ensure!(conv_formats(&model) == conv_formats(&before));
        ensure!(model.nodes().len() == before.nodes().len());
        Ok(())
    }
}

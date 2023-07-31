use tract_hir::internal::*;
use tract_hir::ops::array::{Pad, PadMode};
use tract_hir::ops::cnn::{ConvUnary, PaddingSpec};
use tract_hir::ops::nn::DataFormat;
use tract_hir::tract_core::ops::cnn::KernelFormat;

pub fn rewrite_for_tflite(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for::<ConvUnary>("kernel-in-ohwi", kernel_in_ohwi)
        .with_rule_for::<ConvUnary>("make_1d_2d", make_1d_2d)
        .with_rule_for::<ConvUnary>("force_n_axis", force_n_axis)
        .with_rule_for::<ConvUnary>("nchw-to-nhwc", nchw_to_nhwc)
        .with_rule_for::<ConvUnary>("padding", padding)
        .rewrite(&(), model)
}

fn kernel_in_ohwi(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    let rank = conv.kernel.rank();
    let fact = model.outlet_fact(node.inputs[0])?;
    let shape = conv.pool_spec.data_format.shape(&fact.shape)?;
    if conv.group != 1 && conv.group.to_dim() != *shape.c() {
        bail!("Arbitrary grouping is not supported in tflite")
    }
    let kernel = match conv.kernel_fmt {
        // onnx: (go)IHW => OHW(gi)
        KernelFormat::OIHW => conv
            .kernel
            .clone()
            .into_tensor()
            .split_axis(0, conv.group)?
            .move_axis(0, rank)?
            .move_axis(1, rank)?
            .collapse_axis_with_next(rank - 1),
        // tf: HW(gi)O => OHW(gi)
        KernelFormat::HWIO => conv.kernel.clone().into_tensor().move_axis(rank - 1, 0)?,
        // tflite: (go)HWI
        KernelFormat::OHWI => return Ok(None),
    };
    let mut new = conv.clone();
    new.kernel_fmt = KernelFormat::OHWI;
    new.kernel = kernel.into_arc_tensor();
    let mut patch = TypedModelPatch::default();
    let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
    wire = patch.wire_node(name, new, &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

fn force_n_axis(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if !conv.pool_spec.data_format.has_n() {
        let mut new = conv.clone();
        new.pool_spec.data_format = conv.pool_spec.data_format.with_n();
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.add_n"), AxisOp::Add(0), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_n"), AxisOp::Rm(0), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn make_1d_2d(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if conv.pool_spec.rank() == 1 {
        let pos = conv.pool_spec.data_format.h_axis() + 1;
        let mut new = conv.clone();
        new.pool_spec = conv.pool_spec.change_geo_axes(&AxisOp::Add(1))?;
        let mut kernel = new.kernel.clone().into_tensor();
        kernel.insert_axis(conv.kernel_fmt.h_axis() + 1)?;
        new.kernel = kernel.into_arc_tensor();
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.add_dim"), AxisOp::Add(pos), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_dim"), AxisOp::Rm(pos), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if !conv.pool_spec.data_format.c_is_last() {
        let mut new = conv.clone();
        new.pool_spec.data_format = match conv.pool_spec.data_format {
            DataFormat::NHWC | DataFormat::HWC => unreachable!(),
            DataFormat::CHW => DataFormat::HWC,
            DataFormat::NCHW => DataFormat::NHWC,
        };
        let mut patch = TypedModelPatch::default();
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = conv.pool_spec.data_format.shape(&fact.shape)?;
        let before = shape.c_axis();
        let after = fact.rank() - 1;
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(format!("{name}.nhwc"), AxisOp::Move(before, after), &wire)?;
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.nchw"), AxisOp::Move(after, before), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn padding(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &ConvUnary,
) -> TractResult<Option<TypedModelPatch>> {
    if conv.pool_spec.padding != PaddingSpec::Valid
    // FIXME SameUpper should be usable, but I can't make sense of tflite output
    // && conv.pool_spec.padding != PaddingSpec::SameUpper
    {
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = conv.pool_spec.data_format.shape(&fact.shape)?;
        let actual = conv.pool_spec.computed_padding(shape.hw_dims());
        for pad in [PaddingSpec::Valid /*, PaddingSpec::SameUpper*/] {
            let found = pad.compute(
                shape.hw_dims(),
                &conv.pool_spec.kernel_shape,
                &conv.pool_spec.dilations(),
                &conv.pool_spec.strides(),
            );
            if actual == found {
                let mut new = conv.clone();
                new.pool_spec.padding = pad;
                return Ok(Some(TypedModelPatch::replace_single_op(
                    model,
                    node,
                    &node.inputs,
                    new,
                )?));
            }
        }
        let mut patch = TypedModelPatch::default();
        let mut wires = tvec!(patch.tap_model(model, node.inputs[0])?);
        let mut pads = vec![(0usize, 0usize); fact.rank()];
        for (padding, axis) in actual.iter().zip(shape.hw_axes()) {
            pads[axis] = (padding.pad_before.to_usize()?, padding.pad_after.to_usize()?);
        }
        wires[0] = patch.wire_node(
            format!("{name}.padding"),
            Pad {
                pads,
                mode: PadMode::Constant(Tensor::zero_scalar_dt(fact.datum_type)?.into_arc_tensor()),
            },
            &wires[0..1],
        )?[0];
        let mut new = conv.clone();
        new.pool_spec.padding = PaddingSpec::Valid;
        wires = patch.wire_node(&node.name, new, &wires)?;
        patch.shunt_outside(model, node.id.into(), wires[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

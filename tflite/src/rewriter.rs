use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::cnn::{rewrite_conv_with_n_axis, KernelFormat, MaxPool, PoolSpec, SumPool};
use tract_core::ops::cnn::{Conv, PaddingSpec};
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::Recip;
use tract_core::ops::nn::{expand_mean_of_squares, DataFormat, Softmax};
use tract_core::tract_data::itertools::Itertools;

pub fn rewrite_for_tflite(model: &mut TypedModel) -> TractResult<()> {
    tract_core::ops::einsum::prefix_matmul::rewrite_einsum_to_prefix_matmul(model)?;
    Rewriter::default()
        .with_rule_for("trivial_axes_around_matmul", trivial_axes_around_matmul)
        .with_rule_for("kernel_in_ohwi", kernel_in_ohwi)
        .with_rule_for("bias_as_vector", bias_as_vector)
        //        .with_rule_for("per_layer_in_u8", per_layer_in_u8)
        .with_rule_for("make_1d_2d", make_1d_2d)
        .with_rule_for("rewrite_conv_with_n_axis", rewrite_conv_with_n_axis)
        .with_rule_for("conv-nchw-to-nhwc", conv_nchw_to_nhwc)
        .with_rule_for("maxpool-nchw-to-nhwc", maxpool_nchw_to_nhwc)
        .with_rule_for("sumpool-nchw-to-nhwc", sumpool_nchw_to_nhwc)
        .with_rule_for("padding", padding)
        .with_rule_for("manual_recip", manual_recip)
        .with_rule_for("softmax_on_last_axis", softmax_on_last_axis)
        .with_rule_for("expand-means-of-square", expand_mean_of_squares)
        .rewrite(&(), model)?;
    tract_core::optim::Optimizer::prop_consts().optimize(model)
}

fn trivial_axes_around_matmul(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &PrefixMatMul,
) -> TractResult<Option<TypedModelPatch>> {
    let facts = model.node_input_facts(node.id)?;
    let rank = facts[0].rank();
    if rank <= 4 {
        return Ok(None);
    }
    let trivial_axes = (0..rank - 2)
        .filter(|axis| facts[0].shape[*axis].is_one() && facts[1].shape[*axis].is_one())
        .collect_vec();

    ensure!(!trivial_axes.is_empty(), "Found Einsum with 4 > axes and no trivial axes");
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    for axis in trivial_axes.iter().rev() {
        wire[0] =
            patch.wire_node(format!("{name}.rm_a_axis_{axis}"), AxisOp::Rm(*axis), &[wire[0]])?[0];
        wire[1] =
            patch.wire_node(format!("{name}.rm_b_axis_{axis}"), AxisOp::Rm(*axis), &[wire[1]])?[0];
    }
    let mut out = patch.wire_node(&node.name, *conv, &wire)?;
    for axis in trivial_axes {
        out = patch.wire_node(format!("{name}.add_axis_{axis}"), AxisOp::Add(axis), &out)?;
    }
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

fn kernel_in_ohwi(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    if conv.kernel_fmt == KernelFormat::OHWI {
        return Ok(None);
    }
    if conv.group != 1 && conv.group != conv.output_channels() {
        bail!("Arbitrary grouping is not supported in tflite")
    }
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    let prefix = format!("{name}.kernel_reorg");
    for (ix, op) in conv
        .kernel_fmt
        .kernel_as_group_o_i_h_w_ops(&patch.outlet_fact(wire[1])?.shape, conv.group)
        .into_iter()
        .enumerate()
    {
        wire[1] = patch.wire_node(format!("{prefix}.{ix}"), op, &[wire[1]])?[0];
    }
    let geo_rank = conv.pool_spec.kernel_shape.len();
    // group_o_i_h_w -> o_h_w_gi
    let ci = conv.input_channels();
    wire[1] =
        patch.wire_node(format!("{prefix}.mv_g"), AxisOp::Move(0, geo_rank + 2), &[wire[1]])?[0];
    wire[1] =
        patch.wire_node(format!("{prefix}.mv_i"), AxisOp::Move(1, geo_rank + 2), &[wire[1]])?[0];
    wire[1] = patch.wire_node(
        format!("{prefix}.gi"),
        AxisOp::Reshape(
            geo_rank + 1,
            tvec!(conv.group.to_dim(), (ci / conv.group).to_dim()),
            tvec!(ci.to_dim()),
        ),
        &[wire[1]],
    )?[0];
    let new = Conv { kernel_fmt: KernelFormat::OHWI, ..conv.clone() };
    wire = patch.wire_node(name, new, &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

fn bias_as_vector(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    let bias_fact = model.outlet_fact(node.inputs[2])?;
    let co = conv.output_channels();
    if *bias_fact.shape == [co.to_dim()] {
        return Ok(None);
    }
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    wire[2] = tract_core::ops::cnn::wire_reshape_bias_as_vector(
        &mut patch,
        name,
        wire[2],
        conv.output_channels(),
    )?[0];
    wire = patch.wire_node(name, conv.clone(), &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

/*
fn per_layer_in_u8(
_ctx: &(),
model: &TypedModel,
node: &TypedNode,
name: &str,
conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
let input_fact = model.outlet_fact(node.inputs[0])?;
let idt = input_fact.datum_type;
let kernel_fact = model.outlet_fact(node.inputs[1])?;
let kdt = kernel_fact.datum_type;
if idt.is_float() || model.outlet_fact(node.inputs[6])?.shape.len() > 1 {
return Ok(None);
}
if idt.unquantized() == u8::datum_type() && kdt.unquantized() == u8::datum_type() {
return Ok(None);
}
let mut patch = TypedModelPatch::default();
let wire = patch.taps(model, &node.inputs)?;
let [mut i, mut k, b, mut i0, is, mut k0, ks, o0, os] = &*wire else {
bail!("Unexpected number of inputs")
};
wire_ensure_q8_flavour(&mut patch, name, &mut i, "input", &mut i0, DatumType::U8)?;
wire_ensure_q8_flavour(&mut patch, name, &mut k, "kernel", &mut k0, DatumType::U8)?;
let output = patch.wire_node(name, conv.clone(), &[i, k, *b, i0, *is, k0, *ks, *o0, *os])?;
patch.shunt_outside(model, node.id.into(), output[0])?;
Ok(Some(patch))
}
*/

fn make_1d_2d(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    if conv.pool_spec.rank() == 1 {
        let mut new = conv.clone();
        new.pool_spec = conv.pool_spec.change_geo_axes(&AxisOp::Add(1))?;
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.taps(model, &node.inputs)?;
        let pos_data = conv.pool_spec.data_format.h_axis() + 1;
        wire[0] = patch.wire_node(format!("{name}.add_dim"), AxisOp::Add(pos_data), &[wire[0]])?[0];
        let pos_kernel = conv.kernel_fmt.h_axis() + 1;
        wire[1] =
            patch.wire_node(format!("{name}.add_dim_k"), AxisOp::Add(pos_kernel), &[wire[1]])?[0];
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_dim"), AxisOp::Rm(pos_data), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn conv_nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    nchw_to_nhwc(_ctx, model, node, name, &conv.pool_spec, &|pool_spec| {
        Box::new(Conv { pool_spec, ..conv.clone() })
    })
}

fn maxpool_nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &MaxPool,
) -> TractResult<Option<TypedModelPatch>> {
    nchw_to_nhwc(_ctx, model, node, name, &op.pool_spec, &|pool_spec| {
        Box::new(MaxPool { pool_spec, ..op.clone() })
    })
}

fn sumpool_nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    op: &SumPool,
) -> TractResult<Option<TypedModelPatch>> {
    nchw_to_nhwc(_ctx, model, node, name, &op.pool_spec, &|pool_spec| {
        Box::new(SumPool { pool_spec, ..op.clone() })
    })
}

fn nchw_to_nhwc(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    old: &PoolSpec,
    op: &dyn Fn(PoolSpec) -> Box<dyn TypedOp>,
) -> TractResult<Option<TypedModelPatch>> {
    if !old.data_format.c_is_last() {
        let mut new = old.clone();
        new.data_format = match new.data_format {
            DataFormat::NHWC | DataFormat::HWC => unreachable!(),
            DataFormat::CHW => DataFormat::HWC,
            DataFormat::NCHW => DataFormat::NHWC,
        };
        let mut patch = TypedModelPatch::default();
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = old.data_format.shape(&fact.shape)?;
        let before = shape.c_axis();
        let after = fact.rank() - 1;
        let mut wire = patch.taps(model, &node.inputs)?;
        wire[0] =
            patch.wire_node(format!("{name}.nhwc"), AxisOp::Move(before, after), &[wire[0]])?[0];
        wire = patch.wire_node(name, op(new), &wire)?;
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
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    if conv.pool_spec.padding != PaddingSpec::Valid
    // FIXME SameUpper should be usable, but I can't make sense of tflite output
    // && conv.pool_spec.padding != PaddingSpec::SameUpper
    {
        let fact = model.outlet_fact(node.inputs[0])?;
        let shape = conv.pool_spec.data_format.shape(&fact.shape)?;
        let actual = conv.pool_spec.computed_padding(shape.hw_dims());
        #[allow(clippy::single_element_loop)]
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
        let mut wires = patch.taps(model, &node.inputs)?;
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

fn manual_recip(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    recip: &ElementWiseOp,
) -> TractResult<Option<TypedModelPatch>> {
    if recip.0.is::<Recip>() {
        let mut patch = TypedModelPatch::default();
        let input = patch.tap_model(model, node.inputs[0])?;
        let dt = model.outlet_fact(node.inputs[0])?.datum_type;
        let one = tensor0(1i32).cast_to_dt(dt)?.into_owned().into_tensor();
        let one = patch.add_const(format!("{name}.one"), one)?;
        let wire = wire_with_rank_broadcast(
            name,
            &mut patch,
            tract_core::ops::math::div(),
            &[one, input],
        )?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}

fn softmax_on_last_axis(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    softmax: &Softmax,
) -> TractResult<Option<TypedModelPatch>> {
    let rank = model.outlet_fact(node.inputs[0])?.rank();
    ensure!(softmax.axes.len() == 1);
    if softmax.axes[0] != rank - 1 {
        let mut patch = TypedModelPatch::default();
        let mut wire = tvec!(patch.tap_model(model, node.inputs[0])?);
        wire = patch.wire_node(
            format!("{name}.move_axis"),
            AxisOp::Move(softmax.axes[0], rank - 1),
            &wire,
        )?;
        wire = patch.wire_node(
            format!("{name}.softmax"),
            Softmax { axes: tvec!(rank - 1), ..*softmax },
            &wire,
        )?;
        wire = patch.wire_node(
            format!("{name}.move_axis_back"),
            AxisOp::Move(rank - 1, softmax.axes[0]),
            &wire,
        )?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}

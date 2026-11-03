use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode, Slice};
use tract_core::ops::cnn::{Deconv, wire_reshape_bias_for_bin};
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::add;

/// Kernel size below which the deconvolution stays one kernel launch: the
/// matmul and overlap-add take several, which outweighs the work saved.
const MIN_KERNEL_WEIGHTS: usize = 1 << 14;

/// Rewrite a 1D transposed convolution as a matmul followed by its overlap-add.
///
/// With stride `s` and a kernel zero-padded to `m·s` taps, the matmul produces,
/// for every input frame, the `m·s` output samples it contributes to; tap block
/// `a` of every frame lands `a·s` samples after block 0, so the overlap-add is
/// `m` padded copies summed.  The matmul then runs on the backend's gemm, the
/// rest on copies and adds.
///
/// Output adjustments are the zeros past the overlap-add's last sample, so they
/// become a trailing pad ahead of the padding crop.  Applies to channels-first
/// data with one spatial axis, no dilation, a single group and at least
/// `MIN_KERNEL_WEIGHTS` weights; other deconvolutions are left as they are.
pub fn deconv_as_matmul(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Deconv,
) -> TractResult<Option<TypedModelPatch>> {
    let spec = &op.pool_spec;
    rule_if!(op.group == 1);
    rule_if!(spec.kernel_shape.len() == 1);
    rule_if!(spec.dilations().iter().all(|d| *d == 1));
    rule_if!(!spec.data_format.c_is_last());
    let facts = model.node_input_facts(node.id)?;
    let dt = facts[0].datum_type;
    rule_if!(dt.is_float());
    let x_shape = spec.data_format.shape(facts[0].shape.to_tvec())?;
    let frames = x_shape.hw_dims()[0].clone();
    let (k, s) = (spec.kernel_shape[0], spec.strides()[0]);
    rule_if!(spec.input_channels * spec.output_channels * k >= MIN_KERNEL_WEIGHTS);
    let m = k.div_ceil(s);
    let padded = spec.padding.compute_for_deconv(
        std::slice::from_ref(&frames),
        &spec.kernel_shape,
        &spec.dilations(),
        &spec.strides(),
        &op.adjustments,
    )?;

    let mut patch = TypedModelPatch::default();
    let x = patch.tap_model(model, node.inputs[0])?;
    let mut kernel = tvec!(patch.tap_model(model, node.inputs[1])?);
    let bias = patch.tap_model(model, node.inputs[2])?;
    let kernel_fact = patch.outlet_fact(kernel[0])?.clone();
    for (ix, axis_op) in
        op.kernel_format.kernel_as_group_o_i_hw_ops(&kernel_fact.shape, 1).into_iter().enumerate()
    {
        kernel = patch.wire_node(format!("{node_name}.kernel.{ix}"), axis_op, &kernel)?;
    }
    kernel = patch.wire_node(format!("{node_name}.kernel.rm_g"), AxisOp::Rm(0), &kernel)?;
    if m * s != k {
        let pad = Pad {
            pads: vec![(0, 0), (0, 0), (0, m * s - k)],
            mode: PadMode::Constant(Tensor::zero_scalar_dt(dt)?.into_arc_tensor()),
        };
        kernel = patch.wire_node(format!("{node_name}.kernel.pad_taps"), pad, &kernel)?;
    }

    let has_n = spec.data_format.has_n();
    let (x_axes, cols_axes) = if has_n { ("nit", "nojt") } else { ("it", "ojt") };
    let axes = AxesMapping::from_strs(&["oij", x_axes], &[cols_axes])?;
    let cols = patch.wire_node(
        format!("{node_name}.matmul"),
        EinSum::new(axes, kernel_fact.datum_type),
        &[kernel[0], x],
    )?;

    let tap_axis = has_n as usize + 1;
    let cols = patch.wire_node(
        format!("{node_name}.tap_blocks"),
        AxisOp::Reshape(tap_axis, tvec!((m * s).to_dim()), tvec!(m.to_dim(), s.to_dim())),
        &cols,
    )?[0];
    let mut sum: Option<OutletId> = None;
    for a in 0..m {
        let prefix = format!("{node_name}.block{a}");
        let mut wire =
            patch.wire_node(format!("{prefix}.slice"), Slice::new(tap_axis, a, a + 1), &[cols])?;
        wire = patch.wire_node(format!("{prefix}.rm"), AxisOp::Rm(tap_axis), &wire)?;
        wire = patch.wire_node(
            format!("{prefix}.interleave"),
            AxisOp::Move(tap_axis, tap_axis + 1),
            &wire,
        )?;
        wire = patch.wire_node(
            format!("{prefix}.flatten"),
            AxisOp::Reshape(tap_axis, tvec!(frames.clone(), s.to_dim()), tvec!(frames.clone() * s)),
            &wire,
        )?;
        let mut pads = vec![(0, 0); tap_axis + 1];
        pads[tap_axis] = (a * s, (m - 1 - a) * s);
        let pad =
            Pad { pads, mode: PadMode::Constant(Tensor::zero_scalar_dt(dt)?.into_arc_tensor()) };
        wire = patch.wire_node(format!("{prefix}.shift"), pad, &wire)?;
        sum = Some(match sum {
            None => wire[0],
            Some(acc) => patch.wire_node(format!("{prefix}.add"), add(), &[acc, wire[0]])?[0],
        });
    }
    let mut wire = sum.context("Deconv with an empty kernel")?;

    let full = (frames - 1) * s + m * s;
    let begin = padded[0].pad_before.clone();
    let end = begin.clone() + padded[0].deconvoluted.clone();
    rule_if_some!(past_full = (end.clone() - &full).to_i64().ok());
    if past_full > 0 {
        let mut pads = vec![(0, 0); tap_axis + 1];
        pads[tap_axis] = (0, past_full as usize);
        let pad =
            Pad { pads, mode: PadMode::Constant(Tensor::zero_scalar_dt(dt)?.into_arc_tensor()) };
        wire = patch.wire_node(format!("{node_name}.adjust"), pad, &[wire])?[0];
    }
    if begin != 0.to_dim() || past_full < 0 {
        wire = patch.wire_node(
            format!("{node_name}.crop"),
            Slice::new(tap_axis, begin, end),
            &[wire],
        )?[0];
    }
    let zero_bias = facts[2].konst.as_ref().is_some_and(|b| b.is_zero().unwrap_or(false));
    if zero_bias {
        patch.shunt_outside(model, node.id.into(), wire)?;
        return Ok(Some(patch));
    }
    let bias = wire_reshape_bias_for_bin(
        &mut patch,
        format!("{node_name}.bias"),
        bias,
        tap_axis + 1,
        tap_axis - 1,
        spec.output_channels,
    )?[0];
    wire = patch.wire_node(format!("{node_name}.add_bias"), add(), &[wire, bias])?[0];
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}

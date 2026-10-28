use crate::internal::*;

pub mod conv;
pub mod deconv;
mod maxpool;
mod padding;
mod patch_axis;
mod patches;
pub mod pools;
mod sumpool;

pub use self::conv::{Conv, KernelFormat};
pub use self::deconv::Deconv;
pub use self::maxpool::{MaxPool, OptMaxPool};
pub use self::padding::PaddingSpec;
pub use self::patch_axis::PatchAxis;
pub use self::patches::{Patch, PatchSpec};
pub use self::pools::PoolSpec;
pub use self::sumpool::{OptSumPool, SumPool};

use super::array::MultiBroadcastTo;

pub fn wire_reshape_bias_as_vector(
    model: &mut TypedModel,
    name: impl AsRef<str>,
    outlet: OutletId,
    output_channels: usize,
) -> TractResult<TVec<OutletId>> {
    let name = name.as_ref();
    let mut bias = tvec!(outlet);
    let fact = model.outlet_fact(outlet)?.clone();
    if fact.shape.volume().is_one() && fact.rank() > 0 {
        bias = model.wire_node(
            format!("{name}.bias.make_scalar"),
            AxisOp::Reshape(0, fact.shape.to_tvec(), tvec![]),
            &bias,
        )?;
    }
    if model.outlet_fact(bias[0])?.rank() == 0 {
        bias = model.wire_node(
            format!("{name}.bias.broadcast"),
            MultiBroadcastTo { shape: tvec!(output_channels).into() },
            &bias,
        )?;
    }
    Ok(bias)
}

pub fn wire_reshape_bias_for_bin(
    model: &mut TypedModel,
    name: impl AsRef<str>,
    outlet: OutletId,
    rank: usize,
    c_axis: usize,
    output_channels: usize,
) -> TractResult<TVec<OutletId>> {
    let name = name.as_ref();
    let mut bias = wire_reshape_bias_as_vector(model, name, outlet, output_channels)?;
    let fact = model.outlet_fact(bias[0])?.clone();
    let mut bias_final_shape = tvec![1.to_dim(); rank];
    bias_final_shape[c_axis] = output_channels.to_dim();
    if *bias_final_shape != *fact.shape {
        bias = model.wire_node(
            format!("{name}.bias"),
            AxisOp::Reshape(0, fact.shape.to_tvec(), bias_final_shape),
            &bias,
        )?;
    }
    Ok(bias)
}

pub fn rewrite_conv_with_n_axis(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    if !conv.pool_spec.data_format.has_n() {
        let mut new = conv.clone();
        new.pool_spec.data_format = conv.pool_spec.data_format.with_n();
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.taps(model, &node.inputs)?;
        wire[0] = patch.wire_node(format!("{name}.add_n"), AxisOp::Add(0), &[wire[0]])?[0];
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_n"), AxisOp::Rm(0), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

/// A convolution over one spatial axis as one over two, the second of size one: the input and
/// the kernel each take a unit axis after their spatial one, which the output drops again. For
/// backends whose convolution kernels are two-dimensional only.
pub fn rewrite_conv_1d_to_2d(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    conv: &Conv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(conv.pool_spec.rank() == 1);
    let mut new = conv.clone();
    new.pool_spec = conv.pool_spec.change_geo_axes(&AxisOp::Add(1))?;
    let mut patch = TypedModelPatch::default();
    let mut wire = patch.taps(model, &node.inputs)?;
    let pos_data = conv.pool_spec.data_format.h_axis() + 1;
    wire[0] = patch.wire_node(format!("{name}.add_dim"), AxisOp::Add(pos_data), &[wire[0]])?[0];
    let pos_kernel = conv.kernel_fmt.h_axis() + 1;
    wire[1] = patch.wire_node(format!("{name}.add_dim_k"), AxisOp::Add(pos_kernel), &[wire[1]])?[0];
    wire = patch.wire_node(name, new, &wire)?;
    wire = patch.wire_node(format!("{name}.rm_dim"), AxisOp::Rm(pos_data), &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(Some(patch))
}

pub fn rewrite_deconv_with_n_axis(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    deconv: &Deconv,
) -> TractResult<Option<TypedModelPatch>> {
    if !deconv.pool_spec.data_format.has_n() {
        let mut new = deconv.clone();
        new.pool_spec.data_format = deconv.pool_spec.data_format.with_n();
        let mut patch = TypedModelPatch::default();
        let mut wire = patch.taps(model, &node.inputs)?;
        wire[0] = patch.wire_node(format!("{name}.add_n"), AxisOp::Add(0), &[wire[0]])?[0];
        wire = patch.wire_node(name, new, &wire)?;
        wire = patch.wire_node(format!("{name}.rm_n"), AxisOp::Rm(0), &wire)?;
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

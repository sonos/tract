use super::Downsample;
use crate::internal::*;
use crate::ops;

// trivial cases (sampling on N, mat-mul-as-conv) is handled by invariants
pub fn fuse_downsample_into_conv(
    model: &TypedModel,
    conv_node: &TypedNode,
    conv_op: &ops::cnn::conv::Conv,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    if down_op.stride < 0 {
        return Ok(None);
    }
    let input_fact = model.outlet_fact(conv_node.inputs[0])?;
    let input_shape = conv_op.pool_spec.data_format.shape(input_fact.shape.to_tvec())?;
    if down_op.axis < input_shape.h_axis() {
        return Ok(None);
    }
    let geo_axis = down_op.axis - input_shape.h_axis();
    if geo_axis >= input_shape.rank() {
        return Ok(None);
    }
    let mut new_conv = conv_op.clone();
    if new_conv.pool_spec.strides.is_none() {
        new_conv.pool_spec.strides = Some(tvec!(1; input_shape.hw_rank()));
    }
    new_conv.pool_spec.strides.as_mut().unwrap()[geo_axis] *= down_op.stride as usize;

    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &conv_node.inputs)?;
    let new_output = patch.wire_node(&*conv_node.name, new_conv, &taps)?[0];
    patch.shunt_outside(model, OutletId::new(down_node.id, 0), new_output)?;
    Ok(Some(patch))
}

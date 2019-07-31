use crate::internal::*;
use crate::ops;
use super::Downsample;

// trivial cases (sampling on N, mat-mul-as-conv) is handled by invariants
pub fn fuse_downsample_into_conv(
    model: &TypedModel,
    conv_node: &TypedNode,
    conv_op: &ops::cnn::conv::ConvUnary,
    down_node: &TypedNode,
    down_op: &Downsample,
) -> TractResult<Option<TypedModelPatch>> {
    let input_fact = model.outlet_fact(conv_node.inputs[0])?;
    let input_shape = conv_op.data_format.shape(input_fact.shape.iter().collect::<TVec<_>>());
    if down_op.axis < input_shape.h_axis() {
        return Ok(None)
    }
    let geo_axis = down_op.axis - input_shape.h_axis();
    if geo_axis >= input_shape.rank() {
        return Ok(None)
    }
    let new_conv = conv_op.clone();
    return Ok(None)
        /*
    let geo_axis = full_shape_axis 
    let modulo = (down_op.modulo + crop_op.prune[down_op.axis].0) % down_op.stride;
    let left = (down_op.modulo + crop_op.prune[down_op.axis].0) / down_op.stride;
    let mut patch = TypedModelPatch::default();
    patch.tap_model(model, crop_node.inputs[0])?;
    let input_outlet = crop_node.inputs[0].clone();
    let input_fact = model.outlet_fact(input_outlet).unwrap();
    let final_len = down_node.outputs[0].fact.shape.dim(down_op.axis);
    let new_down = Downsample::new(down_op.axis, down_op.stride, modulo);
    let downed = new_down.transform_fact(&input_fact)?;
    let midway_len = downed.shape.dim(down_op.axis);
    patch.chain(&*down_node.name, new_down, tvec!(downed))?;
    let mut new_prunes = crop_op.prune.clone();
    new_prunes[down_op.axis].0 = left;
    new_prunes[down_op.axis].1 =
        (midway_len.to_dim() - final_len.to_dim() - left).to_integer()? as usize;
    let new_crop = patch.chain(
        &*crop_node.name,
        ops::array::Crop::new(new_prunes),
        tvec!(down_node.outputs[0].fact.clone()),
    )?;
    patch.shunt_outside(OutletId::new(down_node.id, 0), OutletId::new(new_crop, 0))?;
    return Ok(Some(patch));
    */
}

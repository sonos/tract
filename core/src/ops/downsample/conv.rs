use super::Downsample;
use crate::internal::*;
use crate::ops;

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
        return Ok(None);
    }
    let geo_axis = down_op.axis - input_shape.h_axis();
    if geo_axis >= input_shape.rank() {
        return Ok(None);
    }
    let mut new_conv = conv_op.clone();
    new_conv.strides[geo_axis] *= down_op.stride;
    new_conv.full_output_shape[down_op.axis] = down_node.outputs[0].fact.shape.dim(down_op.axis);

    let mut patch = TypedModelPatch::default();
    let tap = patch.tap_model(model, conv_node.inputs[0])?;
    let new_output = patch.wire_node(&*conv_node.name, new_conv, [tap].as_ref())?[0];
    patch.shunt_outside(OutletId::new(down_node.id, 0), new_output)?;
    return Ok(Some(patch));
}

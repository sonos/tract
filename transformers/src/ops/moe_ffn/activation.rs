use crate::ops::gelu_approximate::gelu_approximate;
use crate::ops::silu::silu;
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::math::square;
use tract_nnef::tract_core::ops::nn::leaky_relu;

/// The core ops of an expert activation, applied in order.
pub(super) fn activation_ops(name: &str, has_w3: bool) -> Option<TVec<Box<dyn TypedOp>>> {
    let relu = || -> Box<dyn TypedOp> { Box::new(leaky_relu(0.0)) };
    match name {
        "silu" => Some(tvec!(Box::new(silu()) as _)),
        "swiglu" if has_w3 => Some(tvec!(Box::new(silu()) as _)),
        "gelu" => Some(tvec!(Box::new(gelu_approximate(false)) as _)),
        "relu" => Some(tvec!(relu())),
        "relu2" => Some(tvec!(relu(), Box::new(square()) as _)),
        _ => None,
    }
}

pub(super) fn wire_activation(
    model: &mut TypedModel,
    name: &str,
    ops: TVec<Box<dyn TypedOp>>,
    mut wire: OutletId,
) -> TractResult<OutletId> {
    for (ix, op) in ops.into_iter().enumerate() {
        let node_name = if ix == 0 { name.to_string() } else { format!("{name}.{ix}") };
        wire = model.wire_node(node_name, op, &[wire])?[0];
    }
    Ok(wire)
}

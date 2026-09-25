use crate::ops::gelu_approximate::gelu_approximate;
use crate::ops::silu::silu;
use tract_nnef::internal::*;

pub(super) fn activation_op(name: &str, has_w3: bool) -> Option<Box<dyn TypedOp>> {
    match name {
        "silu" => Some(Box::new(silu())),
        "swiglu" if has_w3 => Some(Box::new(silu())),
        "gelu" => Some(Box::new(gelu_approximate(false))),
        "relu" => Some(Box::new(tract_nnef::tract_core::ops::nn::leaky_relu(0.0))),
        _ => None,
    }
}

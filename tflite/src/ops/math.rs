use crate::registry::Registry;
use crate::tflite::BuiltinOperator;
use tract_hir::tract_core::ops as core;

pub fn register_all(reg: &mut Registry) {
    reg.binary_ops.push((BuiltinOperator::ADD, core::math::add()));
    reg.binary_ops.push((BuiltinOperator::SUB, core::math::sub()));
    reg.binary_ops.push((BuiltinOperator::MUL, core::math::mul()));
    reg.binary_ops.push((BuiltinOperator::DIV, core::math::div()));
}


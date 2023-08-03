use crate::registry::Registry;
use crate::tflite::BuiltinOperator;
use tract_hir::tract_core::ops as core;

pub fn register_all(reg: &mut Registry) {
    reg.reg_binary(BuiltinOperator::ADD, core::math::add());
    reg.reg_binary(BuiltinOperator::SUB, core::math::sub());
    reg.reg_binary(BuiltinOperator::MUL, core::math::mul());
    reg.reg_binary(BuiltinOperator::DIV, core::math::div());
}


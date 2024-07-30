use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    AbsOptions, AbsOptionsArgs, BuiltinOperator, BuiltinOptions, CosOptions, CosOptionsArgs,
    ExpOptions, ExpOptionsArgs, HardSwishOptions, HardSwishOptionsArgs, LeakyReluOptions,
    LeakyReluOptionsArgs, LogicalNotOptionsArgs, SquareOptions, SquareOptionsArgs, LogicalNotOptions,
};
use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::logic::{ Not, not };
use tract_core::ops::math::*;
use tract_core::ops::nn::{hard_swish, leaky_relu, sigmoid, HardSwish, LeakyRelu, Sigmoid};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite(ser);

    reg.reg_to_tract(BuiltinOperator::ABS, |op| deser(op, abs()));
    reg.reg_to_tract(BuiltinOperator::CEIL, |op| deser(op, ceil()));
    reg.reg_to_tract(BuiltinOperator::COS, |op| deser(op, cos()));
    reg.reg_to_tract(BuiltinOperator::EXP, |op| deser(op, exp()));
    reg.reg_to_tract(BuiltinOperator::FLOOR, |op| deser(op, floor()));
    reg.reg_to_tract(BuiltinOperator::HARD_SWISH, |op| deser(op, hard_swish()));
    reg.reg_to_tract(BuiltinOperator::LEAKY_RELU, de_leaky_relu);
    reg.reg_to_tract(BuiltinOperator::LOG, |op| deser(op, ln()));
    reg.reg_to_tract(BuiltinOperator::LOGICAL_NOT, |op| deser(op, not()));
    reg.reg_to_tract(BuiltinOperator::SIN, |op| deser(op, sin()));
    reg.reg_to_tract(BuiltinOperator::LOGISTIC, |op| deser(op, sigmoid()));
    reg.reg_to_tract(BuiltinOperator::SQRT, |op| deser(op, sqrt()));
    reg.reg_to_tract(BuiltinOperator::SQUARE, |op| deser(op, square()));
    reg.reg_to_tract(BuiltinOperator::RSQRT, |op| deser(op, rsqrt()));
    reg.reg_to_tract(BuiltinOperator::TANH, |op| deser(op, tanh()));
}

fn deser(op: &mut DeserOp, ew: ElementWiseOp) -> TractResult<TVec<OutletId>> {
    op.ctx.target.wire_node(op.prefix, ew, op.inputs)
}

fn de_leaky_relu(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_leaky_relu_options);
    op.ctx.target.wire_node(op.prefix, leaky_relu(options.alpha()), op.inputs)
}

fn ser(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &ElementWiseOp,
) -> TractResult<()> {
    let input = builder.map_outlet(model, node.inputs[0])?;
    let output = builder.map_outlet(model, node.id.into())?;
    if (*op.0).is::<Abs>() {
        let options = AbsOptions::create(builder.fb(), &AbsOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(101, 1, BuiltinOperator::ABS, BuiltinOptions::AbsOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<Cos>() {
        let options = CosOptions::create(builder.fb(), &CosOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(108, 1, BuiltinOperator::COS, BuiltinOptions::CosOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<Exp>() {
        let options = ExpOptions::create(builder.fb(), &ExpOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(47, 1, BuiltinOperator::EXP, BuiltinOptions::ExpOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<HardSwish>() {
        let options = HardSwishOptions::create(builder.fb(), &HardSwishOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(117, 1, BuiltinOperator::HARD_SWISH, BuiltinOptions::HardSwishOptions),
            options.as_union_value(),
        )
    } else if let Some(leaky) = (*op.0).downcast_ref::<LeakyRelu>() {
        let options =
            LeakyReluOptions::create(builder.fb(), &LeakyReluOptionsArgs { alpha: leaky.alpha });
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(98, 1, BuiltinOperator::LEAKY_RELU, BuiltinOptions::LeakyReluOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<Not>() {
        let options = LogicalNotOptions::create(builder.fb(), &LogicalNotOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(87, 1, BuiltinOperator::LOGICAL_NOT, BuiltinOptions::LogicalNotOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<Square>() {
        let options = SquareOptions::create(builder.fb(), &SquareOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(92, 1, BuiltinOperator::SQUARE, BuiltinOptions::SquareOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<Ceil>() {
        builder.write_op(&[input], &[output], 104, 1, BuiltinOperator::CEIL)
    } else if (*op.0).is::<Floor>() {
        builder.write_op(&[input], &[output], 8, 1, BuiltinOperator::FLOOR)
    } else if (*op.0).is::<Sin>() {
        builder.write_op(&[input], &[output], 66, 1, BuiltinOperator::SIN)
    } else if (*op.0).is::<Sqrt>() {
        builder.write_op(&[input], &[output], 75, 1, BuiltinOperator::SQRT)
    } else if (*op.0).is::<Rsqrt>() {
        builder.write_op(&[input], &[output], 76, 1, BuiltinOperator::SQRT)
    } else if (*op.0).is::<Sigmoid>() {
        builder.write_op(&[input], &[output], 14, 1, BuiltinOperator::LOGISTIC)
    } else if (*op.0).is::<Tanh>() {
        builder.write_op(&[input], &[output], 28, 1, BuiltinOperator::TANH)
    } else if (*op.0).is::<Ln>() {
        builder.write_op(&[input], &[output], 73, 1, BuiltinOperator::LOG)
    } else {
        todo!("Serialization of ElementWise op {:?}", op)
    }
}

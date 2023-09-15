use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    AbsOptions, AbsOptionsArgs, BuiltinOperator, BuiltinOptions, ExpOptions, ExpOptionsArgs,
    SquareOptions, SquareOptionsArgs,
};
use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::*;

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite(ser);

    reg.reg_to_tract(BuiltinOperator::ABS, |op| deser(op, abs()));
    reg.reg_to_tract(BuiltinOperator::LOG, |op| deser(op, ln()));
    reg.reg_to_tract(BuiltinOperator::EXP, |op| deser(op, exp()));
    reg.reg_to_tract(BuiltinOperator::SQUARE, |op| deser(op, square()));
    reg.reg_to_tract(BuiltinOperator::SQRT, |op| deser(op, sqrt()));
    reg.reg_to_tract(BuiltinOperator::RSQRT, |op| deser(op, rsqrt()));
}

fn deser(op: &mut DeserOp, ew: ElementWiseOp) -> TractResult<TVec<OutletId>> {
    op.ctx.target.wire_node(op.prefix, ew, op.inputs)
}

fn ser(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &ElementWiseOp,
) -> TractResult<()> {
    let input = builder.map_outlet(model, node.inputs[0])?;
    let output = builder.map_outlet(model, node.id.into())?;
    if (*op.0).is::<Square>() {
        let options = SquareOptions::create(builder.fb(), &SquareOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(92, 1, BuiltinOperator::SQUARE, BuiltinOptions::SquareOptions),
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
    } else if (*op.0).is::<Abs>() {
        let options = AbsOptions::create(builder.fb(), &AbsOptionsArgs {});
        builder.write_op_with_options(
            &[input],
            &[output],
            BuiltinOp::new(101, 1, BuiltinOperator::ABS, BuiltinOptions::AbsOptions),
            options.as_union_value(),
        )
    } else if (*op.0).is::<Sqrt>() {
        builder.write_op(&[input], &[output], 75, 1, BuiltinOperator::SQRT)
    } else if (*op.0).is::<Rsqrt>() {
        builder.write_op(&[input], &[output], 76, 1, BuiltinOperator::SQRT)
    } else if (*op.0).is::<Ln>() {
        builder.write_op(&[input], &[output], 73, 1, BuiltinOperator::LOG)
    } else {
        todo!("{:?}", op)
    }
}

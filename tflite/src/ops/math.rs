use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    BuiltinOperator, BuiltinOptions, MaximumMinimumOptions, MaximumMinimumOptionsArgs,
};
use tract_core::internal::*;
use tract_core::ops::binary::{
    wire_cast, wire_rank_broadcast, wire_with_rank_broadcast, TypedBinOp,
};
use tract_core::ops::logic::Less;

pub fn register_all(reg: &mut Registry) {
    /*
    reg.reg_binary(BuiltinOperator::ADD, core::math::add());
    reg.reg_binary(BuiltinOperator::SUB, core::math::sub());
    reg.reg_binary(BuiltinOperator::MUL, core::math::mul());
    reg.reg_binary(BuiltinOperator::DIV, core::math::div());
    reg.reg_binary(BuiltinOperator::MAXIMUM, core::math::max());
    reg.reg_binary(BuiltinOperator::MINIMUM, core::math::min());
    */

    reg.reg_to_tflite(ser_bin);
    reg.reg_to_tract(BuiltinOperator::MAXIMUM, maximum);
    reg.reg_to_tract(BuiltinOperator::MINIMUM, minimum);
}

fn maximum(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let wire = wire_cast(
        &format!("{}.cast", op.prefix),
        op.ctx.target,
        op.inputs,
        DatumType::super_type_for(op.facts()?.iter().map(|f| f.datum_type))
            .context("No super type")?,
    )?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, tract_core::ops::math::max(), &wire)
}

fn minimum(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let wire = wire_cast(
        &format!("{}.cast", op.prefix),
        op.ctx.target,
        op.inputs,
        DatumType::super_type_for(op.facts()?.iter().map(|f| f.datum_type))
            .context("No super type")?,
    )?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, tract_core::ops::math::min(), &wire)
}

fn ser_bin(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &TypedBinOp,
) -> TractResult<()> {
    use tract_linalg::mmm::BinOp;
    let inputs = builder.map_outlets(model, &node.inputs)?;
    let outputs = builder.map_outlets(model, [OutletId::from(node.id)])?;
    if op.0.is::<Less>() {
        builder.write_op(&inputs, &outputs, 58, 1, BuiltinOperator::LESS)
    } else {
        match op.0.as_linalg_binop().unwrap() {
            BinOp::Max => {
                let options =
                    MaximumMinimumOptions::create(builder.fb(), &MaximumMinimumOptionsArgs {});
                builder.write_op_with_options(
                    &inputs,
                    &outputs,
                    BuiltinOp::new(
                        55,
                        1,
                        BuiltinOperator::MAXIMUM,
                        BuiltinOptions::MaximumMinimumOptions,
                    ),
                    options.as_union_value(),
                )
            }
            BinOp::Min => {
                let options =
                    MaximumMinimumOptions::create(builder.fb(), &MaximumMinimumOptionsArgs {});
                builder.write_op_with_options(
                    &inputs,
                    &outputs,
                    BuiltinOp::new(
                        57,
                        1,
                        BuiltinOperator::MINIMUM,
                        BuiltinOptions::MaximumMinimumOptions,
                    ),
                    options.as_union_value(),
                )
            }
            it => todo!("Missing iplementation for binary {it:?} serialization"),
        }
    }
}

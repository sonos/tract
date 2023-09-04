use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    BuiltinOperator, BuiltinOptions, MaximumMinimumOptions, MaximumMinimumOptionsArgs,
};
use tract_core::internal::*;
use tract_core::ops::binary::{wire_cast, wire_with_rank_broadcast, TypedBinOp};
use tract_core::ops::logic;

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite(ser_bin);
    reg.reg_to_tract(BuiltinOperator::MAXIMUM, |op| deser_bin(op, tract_core::ops::math::max()));
    reg.reg_to_tract(BuiltinOperator::MINIMUM, |op| deser_bin(op, tract_core::ops::math::min()));

    reg.reg_to_tract(BuiltinOperator::EQUAL, |op| deser_bin(op, tract_core::ops::logic::equals()));
    reg.reg_to_tract(BuiltinOperator::NOT_EQUAL, |op| deser_bin(op, logic::not_equals()));
    reg.reg_to_tract(BuiltinOperator::LESS, |op| deser_bin(op, tract_core::ops::logic::less()));
    reg.reg_to_tract(BuiltinOperator::LESS_EQUAL, |op| deser_bin(op, logic::less_equal()));
    reg.reg_to_tract(BuiltinOperator::GREATER, |op| deser_bin(op, logic::greater()));
    reg.reg_to_tract(BuiltinOperator::GREATER_EQUAL, |op| deser_bin(op, logic::greater_equal()));
    reg.reg_to_tract(BuiltinOperator::LOGICAL_OR, |op| deser_bin(op, logic::or()));
    reg.reg_to_tract(BuiltinOperator::LOGICAL_AND, |op| deser_bin(op, logic::and()));
}

fn deser_bin(op: &mut DeserOp, mini: TypedBinOp) -> TractResult<TVec<OutletId>> {
    let wire = wire_cast(
        &format!("{}.cast", op.prefix),
        op.ctx.target,
        op.inputs,
        DatumType::super_type_for(op.facts()?.iter().map(|f| f.datum_type))
            .context("No super type")?,
    )?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, mini, &wire)
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

    macro_rules! ser_logic {
        ($tract:ty, $code:expr, $version:expr, $builtin:ident) => {
            if op.0.is::<$tract>() {
                return builder.write_op(
                    &inputs,
                    &outputs,
                    $code,
                    $version,
                    BuiltinOperator::$builtin,
                );
            }
        };
    }
    ser_logic!(logic::Less, 58, 1, LESS);
    ser_logic!(logic::Greater, 61, 1, GREATER);
    ser_logic!(logic::GreaterEqual, 62, 1, GREATER_EQUAL);
    ser_logic!(logic::LessEqual, 63, 1, LESS_EQUAL);
    ser_logic!(logic::Equals, 71, 1, EQUAL);
    ser_logic!(logic::NotEquals, 72, 1, NOT_EQUAL);

    ser_logic!(logic::Or, 84, 1, LOGICAL_OR);
    ser_logic!(logic::And, 86, 1, LOGICAL_AND);

    match op.0.as_linalg_binop().with_context(|| "Missing implementation for binary")? {
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

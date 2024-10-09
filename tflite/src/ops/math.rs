use crate::ops::wire_fused_activation;
use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    ActivationFunctionType, AddOptions, AddOptionsArgs, BuiltinOperator, BuiltinOptions,
    DivOptions, DivOptionsArgs, MaximumMinimumOptions, MaximumMinimumOptionsArgs, MulOptions,
    MulOptionsArgs, SubOptions, SubOptionsArgs,
};
use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::wire_cast;
use tract_core::ops::change_axes::wire_rank_broadcast;
use tract_core::ops::logic::{self, Comp};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite(ser_bin);
    reg.reg_to_tflite(ser_comp);

    reg.reg_to_tract(BuiltinOperator::ADD, deser_add);
    reg.reg_to_tract(BuiltinOperator::SUB, deser_sub);
    reg.reg_to_tract(BuiltinOperator::MUL, deser_mul);
    reg.reg_to_tract(BuiltinOperator::DIV, deser_div);
    reg.reg_to_tract(BuiltinOperator::MAXIMUM, |op| deser_bin(op, tract_core::ops::math::max()));
    reg.reg_to_tract(BuiltinOperator::MINIMUM, |op| deser_bin(op, tract_core::ops::math::min()));

    reg.reg_to_tract(BuiltinOperator::EQUAL, |op| deser_comp(op, Comp::Eq));
    reg.reg_to_tract(BuiltinOperator::NOT_EQUAL, |op| deser_comp(op, Comp::NE));
    reg.reg_to_tract(BuiltinOperator::LESS, |op| deser_comp(op, Comp::LT));
    reg.reg_to_tract(BuiltinOperator::LESS_EQUAL, |op| deser_comp(op, Comp::LTE));
    reg.reg_to_tract(BuiltinOperator::GREATER, |op| deser_comp(op, Comp::GT));
    reg.reg_to_tract(BuiltinOperator::GREATER_EQUAL, |op| deser_comp(op, Comp::GTE));
    reg.reg_to_tract(BuiltinOperator::LOGICAL_OR, |op| deser_bin(op, logic::or()));
    reg.reg_to_tract(BuiltinOperator::LOGICAL_AND, |op| deser_bin(op, logic::and()));
}

fn wire_cast_and_rank_broadcast(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let wire = wire_cast(
        format!("{}.cast", op.prefix),
        op.ctx.target,
        op.inputs,
        DatumType::super_type_for(op.facts()?.iter().map(|f| f.datum_type))
            .context("No super type")?,
    )?;
    wire_rank_broadcast(op.prefix, op.ctx.target, &wire)
}

fn deser_bin(op: &mut DeserOp, mini: TypedBinOp) -> TractResult<TVec<OutletId>> {
    let wires = wire_cast_and_rank_broadcast(op)?;
    op.ctx.target.wire_node(op.prefix, mini, &wires)
}

fn deser_comp(op: &mut DeserOp, comp: Comp) -> TractResult<TVec<OutletId>> {
    let wires = wire_cast_and_rank_broadcast(op)?;
    op.ctx.target.wire_node(op.prefix, comp, &wires)
}

fn deser_add(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_add_options);
    let wires = wire_cast_and_rank_broadcast(op)?;
    let wires = op.ctx.target.wire_node(op.prefix, tract_core::ops::math::add(), &wires)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn deser_sub(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_sub_options);
    let wires = wire_cast_and_rank_broadcast(op)?;
    let wires = op.ctx.target.wire_node(op.prefix, tract_core::ops::math::sub(), &wires)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn deser_mul(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_mul_options);
    let wires = wire_cast_and_rank_broadcast(op)?;
    let wires = op.ctx.target.wire_node(op.prefix, tract_core::ops::math::mul(), &wires)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn deser_div(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_div_options);
    let wires = wire_cast_and_rank_broadcast(op)?;
    let wires = op.ctx.target.wire_node(op.prefix, tract_core::ops::math::div(), &wires)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn ser_bin(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &TypedBinOp,
) -> TractResult<()> {
    use tract_linalg::BinOp;
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

    ser_logic!(logic::Or, 84, 1, LOGICAL_OR);
    ser_logic!(logic::And, 86, 1, LOGICAL_AND);

    if op.0.is::<tract_core::ops::math::Div>() {
        let options = DivOptions::create(
            builder.fb(),
            &DivOptionsArgs { fused_activation_function: ActivationFunctionType::NONE },
        );
        return builder.write_op_with_options(
            &inputs,
            &outputs,
            BuiltinOp::new(42, 1, BuiltinOperator::DIV, BuiltinOptions::DivOptions),
            options.as_union_value(),
        );
    }

    match op.0.as_linalg_binop().with_context(|| "Missing implementation for binary")? {
        BinOp::Add => {
            let options = AddOptions::create(
                builder.fb(),
                &AddOptionsArgs {
                    fused_activation_function: ActivationFunctionType::NONE,
                    pot_scale_int16: false,
                },
            );
            builder.write_op_with_options(
                &inputs,
                &outputs,
                BuiltinOp::new(0, 1, BuiltinOperator::ADD, BuiltinOptions::AddOptions),
                options.as_union_value(),
            )
        }
        BinOp::Sub => {
            let options = SubOptions::create(
                builder.fb(),
                &SubOptionsArgs {
                    fused_activation_function: ActivationFunctionType::NONE,
                    pot_scale_int16: false,
                },
            );
            builder.write_op_with_options(
                &inputs,
                &outputs,
                BuiltinOp::new(41, 1, BuiltinOperator::SUB, BuiltinOptions::SubOptions),
                options.as_union_value(),
            )
        }
        BinOp::Mul => {
            let options = MulOptions::create(
                builder.fb(),
                &MulOptionsArgs { fused_activation_function: ActivationFunctionType::NONE },
            );
            builder.write_op_with_options(
                &inputs,
                &outputs,
                BuiltinOp::new(18, 1, BuiltinOperator::MUL, BuiltinOptions::MulOptions),
                options.as_union_value(),
            )
        }
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

fn ser_comp(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    op: &Comp,
) -> TractResult<()> {
    use Comp::*;
    let (code, version, builtin) = match *op {
        LT => (58, 1, BuiltinOperator::LESS),
        GT => (61, 1, BuiltinOperator::GREATER),
        GTE => (62, 1, BuiltinOperator::GREATER_EQUAL),
        LTE => (63, 1, BuiltinOperator::LESS_EQUAL),
        Eq => (71, 1, BuiltinOperator::EQUAL),
        NE => (72, 1, BuiltinOperator::NOT_EQUAL),
    };
    let inputs = builder.map_outlets(model, &node.inputs)?;
    let outputs = builder.map_outlets(model, [OutletId::from(node.id)])?;
    builder.write_op(&inputs, &outputs, code, version, builtin)
}

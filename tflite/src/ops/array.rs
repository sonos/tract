use tract_hir::internal::*;
use tract_hir::prelude::tract_itertools::Itertools;

use crate::registry::{DeserOp, Registry};
use crate::ser::SubgraphBuilder;
use crate::tflite::{
    BuiltinOperator, BuiltinOptions, ExpandDimsOptions, ExpandDimsOptionsArgs, SqueezeOptions,
    SqueezeOptionsArgs, TransposeOptions, TransposeOptionsArgs,
};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite::<AxisOp>(ser_axisop);
    reg.to_tract.insert(BuiltinOperator::EXPAND_DIMS, de_expand_dims);
    reg.to_tract.insert(BuiltinOperator::RESHAPE, de_reshape);
    reg.to_tract.insert(BuiltinOperator::SQUEEZE, de_squeeze);
    reg.to_tract.insert(BuiltinOperator::TRANSPOSE, de_transpose);
}

fn de_expand_dims(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let axes = op
        .ctx
        .target
        .outlet_fact(op.inputs[1])?
        .konst
        .clone()
        .context("Dynamic EXPAND_DIMS is not supported")?;
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    for (ix, &axis) in axes.as_slice::<i32>()?.iter().sorted().rev().enumerate() {
        wire =
            op.ctx.target.wire_node(format!("{prefix}.{ix}"), AxisOp::Add(axis as usize), &wire)?;
    }
    Ok(wire)
}

fn de_reshape(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input_shape: TVec<TDim> = op.ctx.target.outlet_fact(op.inputs[0])?.shape.to_tvec();
    let shape = op.ctx.target.outlet_fact(op.inputs[1])?.konst.clone().unwrap();
    let shape = shape.cast_to::<TDim>()?;
    let shape = shape.as_slice::<TDim>()?;
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    for (ix, axis_op) in to_axis_ops_with_tf_rules(&input_shape, shape)?.into_iter().enumerate() {
        wire = op.ctx.target.wire_node(format!("{prefix}.{ix}"), axis_op, &wire)?;
    }
    Ok(wire)
}

fn de_squeeze(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_squeeze_options);
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    for (ix, axis) in options.squeeze_dims().unwrap().iter().sorted().enumerate() {
        wire = op.ctx.target.wire_node(format!("{prefix}.{ix}"), AxisOp::Rm(axis as usize), &wire)?;
    }
    Ok(wire)
}

fn de_transpose(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let perm = op
        .ctx
        .target
        .outlet_fact(op.inputs[1])?
        .konst
        .as_ref()
        .context("Dynamic TRANSPOSE in not supported by tract")?;
    let perm = perm.as_slice::<i32>()?.iter().map(|x| *x as usize).collect_vec();
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    for (ix, axis_op) in perm_to_ops(&perm).into_iter().enumerate() {
        wire = op.ctx.target.wire_node(format!("{prefix}.{ix}"), axis_op, &wire)?;
    }
    Ok(wire)
}

fn ser_axisop(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<()> {
    let op = node.op_as::<AxisOp>().unwrap();
    let mut inputs = tvec!(builder.outlets_to_tensors[&node.inputs[0]]);
    let output = builder.outlets_to_tensors[&node.id.into()];
    match op {
        AxisOp::Move(from, to) => {
            let rank = model.node_input_facts(node.id)?[0].rank();
            let mut permutation: Vec<i32> = (0..rank).map(|d| d as i32).collect();
            permutation.remove(*from);
            permutation.insert(*to, *from as _);
            inputs.push(
                builder
                    .write_fact(&format!("{}.perm", node.name), &rctensor1(&permutation).into())?,
            );
            let options = TransposeOptions::create(builder.fb(), &TransposeOptionsArgs {});
            builder.write_op_with_options(
                &inputs,
                &[output],
                BuiltinOperator::TRANSPOSE,
                options.as_union_value(),
                BuiltinOptions::TransposeOptions,
            )
        }
        AxisOp::Add(a) => {
            inputs.push(
                builder.write_fact(&format!("{}.axis", node.name), &rctensor0(*a as i32).into())?,
            );
            let options = ExpandDimsOptions::create(builder.fb(), &ExpandDimsOptionsArgs {});
            builder.write_op_with_options(
                &inputs,
                &[output],
                BuiltinOperator::EXPAND_DIMS,
                options.as_union_value(),
                BuiltinOptions::ExpandDimsOptions,
            )
        }
        AxisOp::Rm(a) => {
            let axes = builder.fb().create_vector(&[*a as i32]);
            let options = SqueezeOptions::create(
                builder.fb(),
                &SqueezeOptionsArgs { squeeze_dims: Some(axes) },
            );
            builder.write_op_with_options(
                &inputs,
                &[output],
                BuiltinOperator::SQUEEZE,
                options.as_union_value(),
                BuiltinOptions::SqueezeOptions,
            )
        }
        _ => todo!("reshape translation"),
    }
}

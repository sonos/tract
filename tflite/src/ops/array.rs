use tract_hir::internal::*;
use tract_hir::ops::array::TypedConcat;
use tract_hir::ops::binary::wire_cast;
use tract_hir::prelude::tract_itertools::Itertools;

use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    BuiltinOperator, BuiltinOptions, ExpandDimsOptions, ExpandDimsOptionsArgs, SqueezeOptions,
    SqueezeOptionsArgs, TransposeOptions, TransposeOptionsArgs,
};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite::<AxisOp>(ser_axisop);
    reg.to_tract.insert(BuiltinOperator::CONCATENATION, de_concat);
    reg.to_tract.insert(BuiltinOperator::EXPAND_DIMS, de_expand_dims);
    reg.to_tract.insert(BuiltinOperator::RESHAPE, de_reshape);
    reg.to_tract.insert(BuiltinOperator::SHAPE, de_shape);
    reg.to_tract.insert(BuiltinOperator::SQUEEZE, de_squeeze);
    reg.to_tract.insert(BuiltinOperator::STRIDED_SLICE, de_strided_slice);
    reg.to_tract.insert(BuiltinOperator::TRANSPOSE, de_transpose);
}

fn de_concat(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_concatenation_options);
    let rank = op.facts()?[0].rank();
    let axis =
        if options.axis() < 0 { rank as i32 + options.axis() } else { options.axis() } as usize;
    let dt = DatumType::super_type_for(op.facts()?.iter().map(|f| f.datum_type)).unwrap();
    let inputs = wire_cast(&op.prefix, &mut op.ctx.target, &op.inputs, dt)?;
    op.ctx.target.wire_node(op.prefix, TypedConcat::new(axis), &inputs)
}

fn de_expand_dims(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, axes) = args_2!(op.facts()?);
    let axes = axes.konst.clone().context("Dynamic EXPAND_DIMS is not supported")?;
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    for (ix, &axis) in axes.as_slice::<i32>()?.iter().sorted().rev().enumerate() {
        let axis = if axis < 0 { axis + input.rank() as i32 } else { axis };
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

fn de_shape(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = args_1!(op.facts()?);
    let wire = op.ctx.target.add_const(op.prefix, tensor1(&input.shape))?;
    Ok(tvec!(wire))
}

fn de_squeeze(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_squeeze_options);
    let mut wire = tvec!(op.inputs[0]);
    let prefix = op.prefix;
    let rank = op.facts()?[0].rank();
    for (ix, axis) in options.squeeze_dims().unwrap().iter().sorted().enumerate() {
        let axis = if axis < 0 { rank as i32 + axis } else { axis } as usize;
        wire =
            op.ctx.target.wire_node(format!("{prefix}.{ix}"), AxisOp::Rm(axis as usize), &wire)?;
    }
    Ok(wire)
}

fn de_strided_slice(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_strided_slice_options);
    ensure!(options.new_axis_mask() == 0 && options.shrink_axis_mask() == 0);
    let slice = tract_core::ops::array::StridedSlice {
        begin_mask: options.begin_mask() as _,
        end_mask: options.end_mask() as _,
        shrink_axis_mask: options.shrink_axis_mask() as _,
        optional_axes_input: None,
        optional_steps_input: Some(3),
    };
    op.ctx.target.wire_node(op.prefix, slice, &op.inputs)
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
                BuiltinOp::new(39, 1, BuiltinOperator::TRANSPOSE, BuiltinOptions::TransposeOptions),
                options.as_union_value(),
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
                BuiltinOp::new(
                    70,
                    1,
                    BuiltinOperator::EXPAND_DIMS,
                    BuiltinOptions::ExpandDimsOptions,
                ),
                options.as_union_value(),
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
                BuiltinOp::new(43, 1, BuiltinOperator::SQUEEZE, BuiltinOptions::SqueezeOptions),
                options.as_union_value(),
            )
        }
        _ => todo!("reshape translation"),
    }
}

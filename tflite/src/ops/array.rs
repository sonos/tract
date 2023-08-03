use tract_hir::internal::*;
use tract_hir::ops::array::TypedConcat;
use tract_hir::ops::binary::wire_cast;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_ndarray::ArrayView2;

use crate::registry::{DeserOp, Registry};
use crate::ser::{BuiltinOp, SubgraphBuilder};
use crate::tflite::{
    BuiltinOperator, BuiltinOptions, ExpandDimsOptions, ExpandDimsOptionsArgs, ReshapeOptions,
    ReshapeOptionsArgs, SqueezeOptions, SqueezeOptionsArgs, TransposeOptions, TransposeOptionsArgs,
};

use super::wire_fused_activation;

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite::<AxisOp>(ser_axisop);
    reg.reg_to_tract(BuiltinOperator::CONCATENATION, de_concat);
    reg.reg_to_tract(BuiltinOperator::EXPAND_DIMS, de_expand_dims);
    reg.reg_to_tract(BuiltinOperator::PADV2, de_padv2);
    reg.reg_to_tract(BuiltinOperator::RESHAPE, de_reshape);
    reg.reg_to_tract(BuiltinOperator::SHAPE, de_shape);
    reg.reg_to_tract(BuiltinOperator::SQUEEZE, de_squeeze);
    reg.reg_to_tract(BuiltinOperator::STRIDED_SLICE, de_strided_slice);
    reg.reg_to_tract(BuiltinOperator::TRANSPOSE, de_transpose);
}

fn de_concat(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let options = builtin!(op, builtin_options_as_concatenation_options);
    let rank = op.facts()?[0].rank();
    let axis =
        if options.axis() < 0 { rank as i32 + options.axis() } else { options.axis() } as usize;
    let dt = DatumType::super_type_for(op.facts()?.iter().map(|f| f.datum_type)).unwrap();
    let inputs = wire_cast(op.prefix, op.ctx.target, op.inputs, dt)?;
    let wires = op.ctx.target.wire_node(op.prefix, TypedConcat::new(axis), &inputs)?;
    wire_fused_activation(op, &wires, &options.fused_activation_function())
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

fn de_padv2(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (_input, pads, value) = args_3!(op.facts()?);
    let pads = pads.konst.as_ref().context("Dynamic PADV2 is not supported")?;
    let prefix = op.prefix;
    let pads: ArrayView2<i32> = pads.to_array_view::<i32>()?.into_dimensionality()?;
    let pads: Vec<(usize, usize)> =
        pads.rows().into_iter().map(|row| (row[0] as usize, row[1] as usize)).collect();
    let mode = tract_hir::ops::array::PadMode::Constant(value.konst.context("Constant expected")?);
    op.ctx.target.wire_node(prefix, tract_core::ops::array::Pad { pads, mode }, &op.inputs[0..1])
}

fn de_reshape(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input_shape: TVec<TDim> = op.ctx.target.outlet_fact(op.inputs[0])?.shape.to_tvec();
    let shape = if let Some(outlet) = op.inputs.get(1) {
        op.ctx.target.outlet_fact(*outlet)?.konst.clone().unwrap()
    } else {
        let options = builtin!(op, builtin_options_as_reshape_options);
        rctensor1(&options.new_shape().as_ref().unwrap().iter().collect::<Vec<i32>>())
    };
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
        wire = op.ctx.target.wire_node(format!("{prefix}.{ix}"), AxisOp::Rm(axis), &wire)?;
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
    op.ctx.target.wire_node(op.prefix, slice, op.inputs)
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
            inputs.push(builder.write_fact(&format!("{}.perm", node.name), tensor1(&permutation))?);
            let options = TransposeOptions::create(builder.fb(), &TransposeOptionsArgs {});
            builder.write_op_with_options(
                &inputs,
                &[output],
                BuiltinOp::new(39, 1, BuiltinOperator::TRANSPOSE, BuiltinOptions::TransposeOptions),
                options.as_union_value(),
            )
        }
        AxisOp::Add(a) => {
            inputs.push(builder.write_fact(&format!("{}.axis", node.name), tensor0(*a as i32))?);
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
        AxisOp::Reshape(_, _, _) => {
            let new_shape = node.outputs[0]
                .fact
                .shape
                .iter()
                .map(|x| x.to_i32())
                .collect::<TractResult<Vec<i32>>>()?;
            let new_shape = builder.fb().create_vector(&new_shape);
            let options = ReshapeOptions::create(
                builder.fb(),
                &ReshapeOptionsArgs { new_shape: Some(new_shape) },
            );
            builder.write_op_with_options(
                &inputs,
                &[output],
                BuiltinOp::new(22, 1, BuiltinOperator::RESHAPE, BuiltinOptions::ReshapeOptions),
                options.as_union_value(),
            )
        }
    }
}

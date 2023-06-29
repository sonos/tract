use tract_hir::internal::*;

use crate::registry::{DeserOp, Registry};
use crate::ser::SubgraphBuilder;
use crate::tflite::{BuiltinOperator, BuiltinOptions, TransposeOptions, TransposeOptionsArgs};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tflite::<AxisOp>(ser_axisop);
    reg.to_tract.insert(BuiltinOperator::RESHAPE, de_reshape);
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
        _ => todo!(),
    }
}

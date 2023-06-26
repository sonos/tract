use tract_hir::internal::*;
use tract_hir::ops::binary::wire_cast;
use tract_hir::tract_core::ops as core;

use crate::registry::{DeserContext, DeserOp, Registry};
use crate::tflite::{ActivationFunctionType, BuiltinOperator};

macro_rules! builtin {
    ($op: expr, $id:ident) => {
        $op.flat.$id().with_context(|| {
            format!(
                "Wrong option type {:?} for operator {:?}",
                $op.flat.builtin_options_type(),
                $op.flat
            )
        })?
    };
}

mod cnn;
mod math;
mod nn;

pub fn register_all(reg: &mut Registry) {
    cnn::register_all(reg);
    math::register_all(reg);
    nn::register_all(reg);

    reg.to_tract.insert(BuiltinOperator::RESHAPE, reshape);
}

fn wire_fused_activation(
    op: &mut DeserOp,
    wires: &[OutletId],
    activation: &ActivationFunctionType,
) -> TractResult<TVec<OutletId>> {
    let prefix = format!("{}.fused", op.prefix);
    let mut op = DeserOp {
        ctx: DeserContext { model: op.ctx.model, subgraph: op.ctx.subgraph, target: op.ctx.target },
        prefix: &prefix,
        flat: op.flat,
        inputs: wires,
    };
    match *activation {
        ActivationFunctionType::NONE => Ok(wires.into()),
        ActivationFunctionType::RELU => nn::relu(&mut op),
        ActivationFunctionType::RELU6 => nn::relu6(&mut op),
        af => bail!("Unsupported fused activation type: {af:?}"),
    }
}

fn reshape(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
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

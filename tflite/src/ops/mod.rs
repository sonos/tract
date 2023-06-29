use tract_hir::internal::*;

use crate::registry::{DeserContext, DeserOp, Registry};
use crate::tflite::ActivationFunctionType;

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

mod array;
mod cnn;
mod math;
mod nn;

pub fn register_all(reg: &mut Registry) {
    array::register_all(reg);
    cnn::register_all(reg);
    math::register_all(reg);
    nn::register_all(reg);
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

use tract_hir::internal::*;
use tract_hir::prelude::tract_itertools::Itertools;

use crate::registry::{DeserContext, DeserOp, Registry};
use crate::tflite::ActivationFunctionType;

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/c/builtin_op_data.h

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
        output_facts: op.output_facts,
    };
    match *activation {
        ActivationFunctionType::NONE => Ok(wires.into()),
        ActivationFunctionType::RELU => nn::de_relu(&mut op),
        ActivationFunctionType::RELU6 => nn::de_relu6(&mut op),
        af => bail!("Unsupported fused activation type: {af:?}"),
    }
}

fn linearops_quantization_suport(
    op: &mut DeserOp,
    input: &TypedFact,
    inputs: &mut TVec<OutletId>,
    kscale_is_per_axis: bool,
) -> TractResult<Option<DatumType>> {
    if op.output_facts[0].datum_type.is_quantized() {
        let p = &op.prefix;
        let iqp = input.datum_type.qparams().unwrap();
        let oqp = op.output_facts[0].datum_type;
        let k_input = op.flat.inputs().unwrap().get(1);
        let k_tensor = op.ctx.subgraph.tensors().unwrap().get(k_input as usize);
        let k_qp = k_tensor.quantization().unwrap();
        let k_scale = if kscale_is_per_axis {
            rctensor1(&k_qp.scale().unwrap().iter().collect_vec())
        } else {
            rctensor0(k_qp.scale().unwrap().get(0))
        };
        let k_zp = k_qp.zero_point().unwrap().iter().map(|i| i as i32).collect_vec();
        ensure!(k_zp.iter().all(|x| *x == 0));
        inputs.push(op.ctx.target.add_const(format!("{p}.k0"), rctensor0(0i8))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.kscale"), k_scale)?);
        inputs.push(op.ctx.target.add_const(format!("{p}.i0"), rctensor0(iqp.zp_scale().0 as i8))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.iscale"), rctensor0(iqp.zp_scale().1))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.c0"), rctensor0(oqp.zp_scale().0 as i8))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.cscale"), rctensor0(oqp.zp_scale().1))?);
        Ok(Some(oqp))
    } else {
        Ok(None)
    }
}

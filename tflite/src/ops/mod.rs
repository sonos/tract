use tract_core::internal::*;
use tract_core::ops::change_axes::wire_with_rank_broadcast;
use tract_core::ops::logic::Iff;
use tract_core::prelude::tract_itertools::Itertools;

use crate::registry::{DeserContext, DeserOp, Registry};
use crate::ser::SubgraphBuilder;
use crate::tflite::{ActivationFunctionType, BuiltinOperator};

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
mod element_wise;
mod math;
mod nn;

pub fn register_all(reg: &mut Registry) {
    array::register_all(reg);
    cnn::register_all(reg);
    element_wise::register_all(reg);
    math::register_all(reg);
    nn::register_all(reg);
    reg.reg_to_tflite(ser_iff);
    reg.reg_to_tract(BuiltinOperator::SELECT, de_iff);
    reg.reg_to_tract(BuiltinOperator::SELECT_V2, de_iff);
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
) -> TractResult<Option<DatumType>> {
    if op.output_facts[0].datum_type.is_quantized() {
        let p = &op.prefix;
        let iqp = input.datum_type.qparams().unwrap();
        let oqp = op.output_facts[0].datum_type;
        let k_input = op.flat.inputs().unwrap().get(1);
        let k_tensor = op.ctx.subgraph.tensors().unwrap().get(k_input as usize);
        let k_qp = k_tensor.quantization().unwrap();
        let k_scale = if k_qp.scale().unwrap().len() > 1 {
            rctensor1(&k_qp.scale().unwrap().iter().collect_vec())
        } else {
            rctensor0(k_qp.scale().unwrap().get(0))
        };
        let k_zp = k_qp.zero_point().unwrap().iter().map(|i| i as i32).collect_vec();
        let k_zp = if k_zp.iter().all_equal() {
            tensor0(k_zp[0])
        } else {
            tensor1(&k_zp)
        };
        inputs.push(op.ctx.target.add_const(format!("{p}.i0"), rctensor0(iqp.zp_scale().0))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.iscale"), rctensor0(iqp.zp_scale().1))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.k0"), k_zp.into_arc_tensor())?);
        inputs.push(op.ctx.target.add_const(format!("{p}.kscale"), k_scale)?);
        inputs.push(op.ctx.target.add_const(format!("{p}.c0"), rctensor0(oqp.zp_scale().0))?);
        inputs.push(op.ctx.target.add_const(format!("{p}.cscale"), rctensor0(oqp.zp_scale().1))?);
        Ok(Some(oqp))
    } else {
        Ok(None)
    }
}

fn ser_iff(
    builder: &mut SubgraphBuilder,
    model: &TypedModel,
    node: &TypedNode,
    _op: &Iff,
) -> TractResult<()> {
    let inputs = builder.map_outlets(model, &node.inputs)?;
    let outputs = builder.map_outlets(model, [OutletId::new(node.id, 0)])?;
    builder.write_op(&inputs, &outputs, 123, 1, BuiltinOperator::SELECT_V2)
}

fn de_iff(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    wire_with_rank_broadcast(op.prefix, op.ctx.target, Iff, op.inputs)
}

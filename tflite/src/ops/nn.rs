use tract_hir::internal::*;
use tract_hir::ops::binary::wire_cast;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::prelude::tract_itertools::Itertools;
use tract_hir::tract_core::ops as core;
use tract_hir::tract_core::ops::cast::Cast;
use tract_hir::tract_core::ops::einsum::EinSum;

use crate::registry::{DeserOp, Registry};
use crate::tflite::{BuiltinOperator, FullyConnectedOptionsWeightsFormat};

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tract(BuiltinOperator::FULLY_CONNECTED, de_fully_connected);
    reg.reg_to_tract(BuiltinOperator::MEAN, de_reduce_mean);
    reg.reg_to_tract(BuiltinOperator::SOFTMAX, de_softmax);

    reg.reg_to_tract(BuiltinOperator::RELU, de_relu);
    reg.reg_to_tract(BuiltinOperator::RELU6, de_relu6);
    reg.reg_element_wise(BuiltinOperator::HARD_SWISH, Box::new(core::nn::HardSwish {}));
}

fn de_fully_connected(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, _weights, _bias) = args_3!(op.facts()?);
    let options = builtin!(op, builtin_options_as_fully_connected_options);
    ensure!(input.datum_type.is_quantized());
    ensure!(options.weights_format() == FullyConnectedOptionsWeightsFormat::DEFAULT);
    ensure!(!options.keep_num_dims());
    ensure!(!options.asymmetric_quantize_inputs());
    let mut inputs: TVec<OutletId> = op.inputs.into();
    let qp = super::linearops_quantization_suport(
        op,
        &input,
        &mut inputs,
        false,
    )?;
    let operating_dt =
        if input.datum_type.is_float() { input.datum_type } else { i32::datum_type() };
    let einsum = EinSum { axes: "BI,OI,I,,,,,,->BO".parse()?, q_params: qp, operating_dt };
    let wires = op.ctx.target.wire_node(op.prefix, einsum, &inputs)?;
    super::wire_fused_activation(op, &wires, &options.fused_activation_function())
}

fn de_reduce_mean(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, axes) = args_2!(op.facts()?);
    let options = builtin!(op, builtin_options_as_reducer_options);
    let axes: TVec<usize> = axes
        .konst
        .as_ref()
        .unwrap()
        .as_slice::<i32>()?
        .iter()
        .map(|d| *d as usize)
        .sorted()
        .collect();
    let norm: TDim = axes.iter().map(|d| &input.shape[*d]).product();
    let p = &op.prefix;
    let mut wire = op.ctx.target.wire_node(
        format!("{p}.sum"),
        core::nn::Reduce::new(axes.clone(), core::nn::Reducer::Sum),
        &[op.inputs[0]],
    )?;
    if !options.keep_dims() {
        for axis in axes.iter().rev() {
            wire =
                op.ctx.target.wire_node(format!("{p}.rm_axis_{axis}"), AxisOp::Rm(*axis), &wire)?;
        }
    }
    let norm = op.ctx.target.add_const(format!("{p}.card"), tensor0(norm))?;
    let norm = op.ctx.target.wire_node(
        format!("{p}.as_float"),
        Cast { to: f32::datum_type() },
        &[norm],
    )?;
    let norm = op.ctx.target.wire_node(format!("{p}.recip"), core::math::recip(), &norm)?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, core::quant::scale(), &[norm[0], wire[0]])
}

fn de_softmax(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = args_1!(op.facts()?);
    let options = builtin!(op, builtin_options_as_softmax_options);
    ensure!(options.beta() == 1.0);
    let softmax = core::nn::Softmax { axes: tvec!(input.rank() - 1), output_dt: input.datum_type };
    op.ctx.target.wire_node(op.prefix, softmax, op.inputs)
}

pub fn de_relu(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = op.inputs[0];
    let zero = op.ctx.target.add_const(format!("{}.zero", op.prefix), tensor0(0f32))?;
    let wires = wire_cast(
        op.prefix,
        op.ctx.target,
        &[input, zero],
        op.ctx.target.outlet_fact(input)?.datum_type,
    )?;
    wire_with_rank_broadcast(
        &format!("{}.relu", op.prefix),
        op.ctx.target,
        core::math::max(),
        &wires,
    )
}

pub fn de_relu6(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = de_relu(op)?[0];
    let six = op.ctx.target.add_const(format!("{}.six", op.prefix), tensor0(6f32))?;
    let wires = wire_cast(
        op.prefix,
        op.ctx.target,
        &[input, six],
        op.ctx.target.outlet_fact(input)?.datum_type,
    )?;
    wire_with_rank_broadcast(
        &format!("{}.relu6", op.prefix),
        op.ctx.target,
        core::math::min(),
        &wires,
    )
}

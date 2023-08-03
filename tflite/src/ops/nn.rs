use tract_hir::internal::*;
use tract_hir::ops::binary::wire_cast;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_hir::tract_core::ops as core;

use crate::registry::{DeserOp, Registry};
use crate::tflite::BuiltinOperator;

pub fn register_all(reg: &mut Registry) {
    reg.reg_to_tract(BuiltinOperator::MEAN, reduce_mean);
    reg.reg_to_tract(BuiltinOperator::SOFTMAX, softmax);

    reg.reg_to_tract(BuiltinOperator::RELU, relu);
    reg.reg_to_tract(BuiltinOperator::RELU6, relu6);
    reg.reg_element_wise(BuiltinOperator::HARD_SWISH, Box::new(core::nn::HardSwish {}));
}

fn reduce_mean(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let (input, axes) = args_2!(op.facts()?);
    let options = builtin!(op, builtin_options_as_reducer_options);
    ensure!(options.keep_dims());
    let axes: TVec<usize> =
        axes.konst.as_ref().unwrap().as_slice::<i32>()?.iter().map(|d| *d as usize).collect();
    let norm: TDim = axes.iter().map(|d| &input.shape[*d]).product();
    let wire = op.ctx.target.wire_node(
        op.prefix.to_string() + ".sum",
        core::nn::Reduce::new(axes, core::nn::Reducer::Sum),
        &[op.inputs[0]],
    )?;
    let norm = op.ctx.target.add_const("{prefix}.card", tensor0(norm))?;
    let wires = wire_cast(op.prefix, op.ctx.target, &[wire[0], norm], input.datum_type)?;
    wire_with_rank_broadcast(op.prefix, op.ctx.target, core::math::div(), &wires)
}

fn softmax(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = args_1!(op.facts()?);
    let options = builtin!(op, builtin_options_as_softmax_options);
    ensure!(options.beta() == 1.0);
    let softmax = core::nn::Softmax { axes: tvec!(input.rank() - 1), output_dt: input.datum_type };
    op.ctx.target.wire_node(op.prefix, softmax, op.inputs)
}

pub fn relu(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
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

pub fn relu6(op: &mut DeserOp) -> TractResult<TVec<OutletId>> {
    let input = relu(op)?[0];
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

use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;

mod array;
mod cast;
pub mod cumsum;
mod d2s;
mod einsum;
mod fft;
pub mod logic;
mod math;
mod ml;
pub mod multinomial;
mod nn;
mod non_max_suppression;
mod quant;
mod random;
pub mod rec;
mod resize;
mod s2d;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Constant", konst);
    reg.insert("Einsum", einsum::einsum);
    reg.insert("Identity", |_, _| {
        Ok((Box::<tract_hir::ops::identity::Identity>::default(), vec![]))
    });
    reg.insert("Resize", resize::resize);
    reg.insert("NonMaxSuppression", non_max_suppression::non_max_suppression);
    reg.insert("Multinomial", multinomial::multinomial);
    array::register_all_ops(reg);
    cast::register_all_ops(reg);
    cumsum::register_all_ops(reg);
    d2s::register_all_ops(reg);
    fft::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    ml::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    random::register_all_ops(reg);
    rec::register_all_ops(reg);
    s2d::register_all_ops(reg);
}

fn konst(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let value = if let Some(v) = node.get_attr_opt("value")? {
        ctx.load_tensor(v)?
    } else if let Some(i) = node.get_attr_opt::<i64>("value_int")? {
        tensor0(i)
    } else if let Some(v) = node.get_attr_opt::<f32>("value_float")? {
        tensor0(v)
    } else {
        bail!("Could not extract value out of Constant node")
    };
    Ok((Box::new(tract_hir::ops::konst::Const::new(value.into_arc_tensor())?), vec![]))
}

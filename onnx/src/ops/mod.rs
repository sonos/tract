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
    } else if let Some(ints) = node.get_attr_opt_slice::<i64>("value_ints")? {
        tensor1(ints)
    } else if let Some(floats) = node.get_attr_opt_slice::<f32>("value_floats")? {
        tensor1(floats)
    } else if let Some(s) = node.get_attr_opt::<String>("value_string")? {
        tensor0(s)
    } else if let Some(strings) = node.get_attr_opt_slice::<Vec<u8>>("value_strings")? {
        let strings: Vec<String> =
            strings.iter().map(|b| String::from_utf8_lossy(b).into_owned()).collect();
        tensor1(&strings)
    } else {
        bail!("Could not extract value out of Constant node")
    };
    Ok((Box::new(tract_hir::ops::konst::Const::new(value.into_arc_tensor())?), vec![]))
}

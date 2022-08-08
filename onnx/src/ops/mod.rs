use crate::model::{OnnxOpRegister, ParsingContext};
use crate::pb::*;
use tract_hir::internal::*;
use tract_hir::ops;

macro_rules! op_onnx {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["onnx"]
        }
    };
}

mod array;
mod cast;
mod cumsum;
mod d2s;
mod einsum;
mod logic;
mod math;
mod ml;
mod nn;
mod quant;
pub mod rec;
mod resize;
mod non_max_suppression;
mod multinomial;
mod s2d;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Cast", cast::cast);
    reg.insert("Constant", konst);
    reg.insert("Einsum", einsum::einsum);
    reg.insert("Identity", |_, _| Ok((Box::new(ops::identity::Identity::default()), vec![])));
    reg.insert("Resize", resize::resize);
    reg.insert("NonMaxSuppression", non_max_suppression::non_max_suppression);
    reg.insert("Multinomial", multinomial::multinomial);
    array::register_all_ops(reg);
    cumsum::register_all_ops(reg);
    d2s::register_all_ops(reg);
    logic::register_all_ops(reg);
    math::register_all_ops(reg);
    ml::register_all_ops(reg);
    nn::register_all_ops(reg);
    quant::register_all_ops(reg);
    rec::register_all_ops(reg);
    s2d::register_all_ops(reg);
}

fn konst(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let v = node.get_attr::<Tensor>("value")?;
    Ok((Box::new(tract_hir::ops::konst::Const(v.into())), vec![]))
}

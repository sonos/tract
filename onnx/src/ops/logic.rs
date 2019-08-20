use crate::model::OnnxOpRegister;
use tract_core::ops as tractops;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(tractops::logic::Not::default()),vec!())));
    reg.insert("And", |_, _| Ok((Box::new(tractops::logic::and()),vec!())));
    reg.insert("Or", |_, _| Ok((Box::new(tractops::logic::or()),vec!())));
    reg.insert("Xor", |_, _| Ok((Box::new(tractops::logic::xor()),vec!())));

    reg.insert("Equal", |_, _| Ok((Box::new(tractops::logic::equals()),vec!())));
    reg.insert("Greater", |_, _| Ok((Box::new(tractops::logic::greater()),vec!())));
    reg.insert("Less", |_, _| Ok((Box::new(tractops::logic::lesser()),vec!())));

    reg.insert("Where", |_, _| Ok((Box::new(tractops::logic::Iff::default()),vec!())));
}

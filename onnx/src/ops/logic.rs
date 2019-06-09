use crate::model::OnnxOpRegister;
use tract_core::ops as tractops;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(tractops::logic::Not::default()),vec!())));
    reg.insert("And", |_, _| Ok((Box::new(tractops::logic::And::default()),vec!())));
    reg.insert("Or", |_, _| Ok((Box::new(tractops::logic::Or::default()),vec!())));
    reg.insert("Xor", |_, _| Ok((Box::new(tractops::logic::Xor::default()),vec!())));

    reg.insert("Equal", |_, _| Ok((Box::new(tractops::logic::Equals::default()),vec!())));
    reg.insert("Greater", |_, _| Ok((Box::new(tractops::logic::Greater::default()),vec!())));
    reg.insert("Less", |_, _| Ok((Box::new(tractops::logic::Lesser::default()),vec!())));

    reg.insert("Where", |_, _| Ok((Box::new(tractops::logic::Iff::default()),vec!())));
}

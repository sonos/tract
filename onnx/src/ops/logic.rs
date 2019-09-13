use crate::model::OnnxOpRegister;
use tract_core::ops as tractops;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(tractops::logic::not()), vec![])));
    reg.insert("And", |_, _| Ok((Box::new(tractops::logic::and::bin()), vec![])));
    reg.insert("Or", |_, _| Ok((Box::new(tractops::logic::or::bin()), vec![])));
    reg.insert("Xor", |_, _| Ok((Box::new(tractops::logic::xor::bin()), vec![])));

    reg.insert("Equal", |_, _| Ok((Box::new(tractops::logic::equals::bin()), vec![])));
    reg.insert("Greater", |_, _| Ok((Box::new(tractops::logic::greater::bin()), vec![])));
    reg.insert("Less", |_, _| Ok((Box::new(tractops::logic::lesser::bin()), vec![])));

    reg.insert("Where", |_, _| Ok((Box::new(tractops::logic::Iff::default()), vec![])));
}

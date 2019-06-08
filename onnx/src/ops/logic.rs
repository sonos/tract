use crate::model::OnnxOpRegister;
use tract_core::ops as tractops;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok(Box::new(tractops::logic::Not::default())));
    reg.insert("And", |_, _| Ok(Box::new(tractops::logic::And::default())));
    reg.insert("Or", |_, _| Ok(Box::new(tractops::logic::Or::default())));
    reg.insert("Xor", |_, _| Ok(Box::new(tractops::logic::Xor::default())));

    reg.insert("Equal", |_, _| Ok(Box::new(tractops::logic::Equals::default())));
    reg.insert("Greater", |_, _| Ok(Box::new(tractops::logic::Greater::default())));
    reg.insert("Less", |_, _| Ok(Box::new(tractops::logic::Lesser::default())));
}

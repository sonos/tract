use tract_core::ops as tractops;
use tract_core::ops::prelude::*;
use crate::pb::NodeProto;

pub fn register_all_ops(reg: &mut OpRegister<NodeProto>) {
    reg.insert("Not", |_| Ok(Box::new(tractops::logic::Not::default())));
    reg.insert("And", |_| Ok(Box::new(tractops::logic::And::default())));
    reg.insert("Or", |_| Ok(Box::new(tractops::logic::Or::default())));
    reg.insert("Xor", |_| Ok(Box::new(tractops::logic::Xor::default())));

    reg.insert("Equal", |_| {
        Ok(Box::new(tractops::logic::Equals::default()))
    });
    reg.insert("Greater", |_| {
        Ok(Box::new(tractops::logic::Greater::default()))
    });
    reg.insert("Less", |_| Ok(Box::new(tractops::logic::Lesser::default())));
}

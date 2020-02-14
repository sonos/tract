use crate::model::OnnxOpRegister;
use tract_hir::ops;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(ops::logic::not()), vec![])));
    reg.insert("And", |_, _| Ok((Box::new(ops::logic::and::bin()), vec![])));
    reg.insert("Or", |_, _| Ok((Box::new(ops::logic::or::bin()), vec![])));
    reg.insert("Xor", |_, _| Ok((Box::new(ops::logic::xor::bin()), vec![])));

    reg.insert("Equal", |_, _| Ok((Box::new(ops::logic::equals::bin()), vec![])));
    reg.insert("Greater", |_, _| Ok((Box::new(ops::logic::greater::bin()), vec![])));
    reg.insert("Less", |_, _| Ok((Box::new(ops::logic::lesser::bin()), vec![])));

    reg.insert("Where", |_, _| Ok((Box::new(ops::logic::Iff::default()), vec![])));
}

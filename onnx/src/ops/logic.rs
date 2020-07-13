use crate::model::OnnxOpRegister;
use tract_hir::internal::*;
use tract_hir::ops;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("Not", |_, _| Ok((Box::new(ops::logic::not()), vec![])));
    reg.insert("And", |_, _| Ok((ops::logic::And.into_hir(), vec![])));
    reg.insert("Or", |_, _| Ok((ops::logic::Or.into_hir(), vec![])));
    reg.insert("Xor", |_, _| Ok((ops::logic::Xor.into_hir(), vec![])));

    reg.insert("Equal", |_, _| Ok((ops::logic::Equals.into_hir(), vec![])));
    reg.insert("Greater", |_, _| Ok((ops::logic::Greater.into_hir(), vec![])));
    reg.insert("Less", |_, _| Ok((ops::logic::Lesser.into_hir(), vec![])));

    reg.insert("Where", |_, _| Ok((Box::new(ops::logic::Iff::default()), vec![])));
}

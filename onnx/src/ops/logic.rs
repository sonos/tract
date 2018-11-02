use tract::ops as tfdops;

use ops::OpRegister;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Not", |_| Ok(Box::new(tfdops::logic::Not::default())));
    reg.insert("And", |_| Ok(Box::new(tfdops::logic::And::default())));
    reg.insert("Or", |_| Ok(Box::new(tfdops::logic::Or::default())));
    reg.insert("Xor", |_| Ok(Box::new(tfdops::logic::Xor::default())));

    reg.insert("Equal", |_| Ok(Box::new(tfdops::logic::Equals::default())));
    reg.insert("Greater", |_| {
        Ok(Box::new(tfdops::logic::Greater::default()))
    });
    reg.insert("Less", |_| Ok(Box::new(tfdops::logic::Lesser::default())));
}

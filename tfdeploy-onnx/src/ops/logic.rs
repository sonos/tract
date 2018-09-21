use tfdeploy::ops as tfdops;

use pb::NodeProto;
use ops::OpRegister;
use tfdeploy::TfdResult;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Not", |_| Ok(Box::new(tfdops::logic::Not::default())));
    reg.insert("And", |_| Ok(Box::new(tfdops::logic::And::default())));
    reg.insert("Or", |_| Ok(Box::new(tfdops::logic::Or::default())));
    reg.insert("Xor", |_| Ok(Box::new(tfdops::logic::Xor::default())));
}

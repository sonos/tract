use tfdeploy::ops as tfdops;

use pb::NodeProto;
use ops::OpRegister;
use tfdeploy::TfdResult;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Add", |_| Ok(Box::new(tfdops::math::Add::default())));
    reg.insert("Sub", |_| Ok(Box::new(tfdops::math::Sub::default())));
    reg.insert("Mul", |_| Ok(Box::new(tfdops::math::Mul::default())));
    reg.insert("Div", |_| Ok(Box::new(tfdops::math::Div::default())));

    reg.insert("Sum", |_| Ok(Box::new(tfdops::math::add_n::AddN::default())));

    reg.insert("Abs", |_| Ok(Box::new(tfdops::math::Abs::default())));
    reg.insert("Ceil", |_| Ok(Box::new(tfdops::math::Ceil::default())));
    reg.insert("Floor", |_| Ok(Box::new(tfdops::math::Floor::default())));

    reg.insert("Cos", |_| Ok(Box::new(tfdops::math::Cos::default())));
    reg.insert("Sin", |_| Ok(Box::new(tfdops::math::Sin::default())));
    reg.insert("Tan", |_| Ok(Box::new(tfdops::math::Tan::default())));
    reg.insert("Acos", |_| Ok(Box::new(tfdops::math::Acos::default())));
    reg.insert("Asin", |_| Ok(Box::new(tfdops::math::Asin::default())));
    reg.insert("Atan", |_| Ok(Box::new(tfdops::math::Atan::default())));

    reg.insert("Tanh", |_| Ok(Box::new(tfdops::math::Tanh::default())));
}


use tfdeploy::ops as tfdops;

use ops::OpRegister;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("Expand", |_| Ok(Box::new(tfdops::array::MultiBroadcastTo::default())));
}

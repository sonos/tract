mod philox;
mod random_uniform;

use crate::model::TfOpRegister;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("RandomUniform", random_uniform::random_uniform);
    reg.insert("RandomUniformInt", random_uniform::random_uniform_int);
}

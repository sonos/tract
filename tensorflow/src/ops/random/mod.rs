mod philox;
mod random_uniform;

use crate::model::TfOpRegister;
use crate::tfpb::node_def::NodeDef;


pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("RandomUniform", random_uniform::random_uniform);
}

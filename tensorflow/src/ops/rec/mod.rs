use crate::model::TfOpRegister;

pub mod block_lstm;

pub fn register_all_ops(reg: &mut TfOpRegister) {
    reg.insert("BlockLSTM", block_lstm::block_lstm);
}

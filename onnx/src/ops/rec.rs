use crate::model::OnnxOpRegister;

pub mod common;
pub mod gru;
pub mod lstm;
pub mod rnn;
pub mod scan;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("GRU", gru::gru);
    reg.insert("LSTM", lstm::lstm);
    reg.insert("RNN", rnn::rnn);
    reg.insert("Scan", scan::scan);
}

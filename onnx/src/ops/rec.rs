use crate::model::OnnxOpRegister;

pub mod lstm;
pub mod rnn;
pub mod scan;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LSTM", lstm::lstm);
    reg.insert("RNN", rnn::rnn);
    reg.insert("Scan", scan::scan);
}

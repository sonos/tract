use crate::model::OnnxOpRegister;

pub mod lstm;
// pub mod scan;

pub fn register_all_ops(reg: &mut OnnxOpRegister) {
    reg.insert("LSTM", lstm::lstm);
//    reg.insert("Scan", scan::scan);
}

use crate::ops::OpRegister;

mod lstm;

pub fn register_all_ops(reg: &mut OpRegister) {
    reg.insert("LSTM", lstm::lstm);
}

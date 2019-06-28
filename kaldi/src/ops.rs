use crate::model::KaldiOpRegister;

mod affine;

pub fn register_all_ops(reg: &mut KaldiOpRegister) {
    reg.insert("FixedAffineComponent", affine::fixed_affine_component);
}

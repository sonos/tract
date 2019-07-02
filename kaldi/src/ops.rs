use crate::model::KaldiOpRegister;

mod affine;
mod renorm;

pub fn register_all_ops(reg: &mut KaldiOpRegister) {
    reg.insert("FixedAffineComponent", affine::affine_component);
    reg.insert("NaturalGradientAffineComponent", affine::affine_component);
    reg.insert("RectifiedLinearComponent", |_,_| Ok(Box::new(tract_core::ops::nn::Relu::default())));
    reg.insert("NormalizeComponent", renorm::renorm);
}

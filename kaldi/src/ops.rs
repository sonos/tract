use crate::model::KaldiOpRegister;

pub(crate) mod affine;
pub(crate) mod lstm_nonlin;
pub(crate) mod memory;
mod renorm;

pub const AFFINE: &'static [&'static str] =
    &["FixedAffineComponent", "NaturalGradientAffineComponent"];

pub fn register_all_ops(reg: &mut KaldiOpRegister) {
    for affine in AFFINE {
        reg.insert(affine, affine::affine_component);
    }
    reg.insert("BackpropTruncationComponent", |_, _| {
        Ok(Box::new(tract_core::ops::identity::Identity::default()))
    });
    reg.insert("NormalizeComponent", renorm::renorm);
    reg.insert("LstmNonlinearityComponent", lstm_nonlin::lstm_nonlin);
    reg.insert("RectifiedLinearComponent", |_, _| {
        Ok(Box::new(tract_core::ops::math::scalar_max((0.0).into())))
    });
}

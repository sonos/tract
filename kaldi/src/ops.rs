use crate::model::KaldiOpRegister;
use tract_hir::internal::*;

#[macro_export]
macro_rules! op_kaldi {
    () => {
        fn op_families(&self) -> &'static [&'static str] {
            &["core"]
        }
    };
}

pub(crate) mod affine;
pub(crate) mod lstm_nonlin;
pub(crate) mod memory;
mod renorm;

pub const AFFINE: &[&str] = &["FixedAffineComponent", "NaturalGradientAffineComponent"];

pub fn register_all_ops(reg: &mut KaldiOpRegister) {
    for affine in AFFINE {
        reg.insert(affine, affine::affine_component);
    }
    reg.insert("BackpropTruncationComponent", |_, _| {
        Ok(Box::new(tract_hir::ops::identity::Identity::default()))
    });
    reg.insert("NormalizeComponent", renorm::renorm);
    reg.insert("LstmNonlinearityComponent", lstm_nonlin::lstm_nonlin);
    reg.insert("RectifiedLinearComponent", |_, _| {
        Ok(expand(tract_hir::ops::activations::Clip::new(Some(0.0), None)))
    });
}

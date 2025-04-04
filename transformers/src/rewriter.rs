use std::borrow::Cow;

use tract_nnef::prelude::{Rewriter, TractResult, TypedModel};
use tract_nnef::tract_core::transform::ModelTransform;

use crate::rewrite_rules;

#[derive(Debug, Default)]
pub struct RmsNormTransform;

impl ModelTransform for RmsNormTransform {
    fn name(&self) -> Cow<str> {
        "rms-norm-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
        .with_rule_for("as-rms-norm", rewrite_rules::as_rms_norm_rule)
        .with_rule_for("remove_rms_norm_cast", rewrite_rules::remove_rms_norm_cast)
        .rewrite(&(), model)
    }
}
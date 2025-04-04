use std::borrow::Cow;

use tract_nnef::prelude::{Rewriter, TractResult, TypedModel};
use tract_nnef::tract_core::transform::ModelTransform;

use crate::ops;

#[derive(Debug, Default)]
pub struct RmsNormTransform;

impl ModelTransform for RmsNormTransform {
    fn name(&self) -> Cow<str> {
        "rms-norm-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
        .with_rule_for("as-rms-norm", ops::as_rms_norm_rule)
        .with_rule_for("remove_rms_norm_cast", ops::remove_rms_norm_cast)
        .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct ApplyRopeTransform;

impl ModelTransform for ApplyRopeTransform {
    fn name(&self) -> Cow<str> {
        "apply-rope-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
        .with_rule_for("as-rotate-half", ops::as_rotate_half_rule)
        .with_rule_for("as-apply-rope", ops::as_apply_rope_rule)
        .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct SiluTransform;

impl ModelTransform for SiluTransform {
    fn name(&self) -> Cow<str> {
        "silu-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
        .with_rule_for("as-silu", ops::as_silu_rule)
        .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct ScaledMaskedSoftmaxTransform;

impl ModelTransform for ScaledMaskedSoftmaxTransform {
    fn name(&self) -> Cow<str> {
        "silu-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
        .with_rule_for("as-scaled-masked-softmax", ops::as_scaled_masked_softmax_rule)
        .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct NewGeluTransform;

impl ModelTransform for NewGeluTransform {
    fn name(&self) -> Cow<str> {
        "silu-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
        .with_rule_for("as-new-gelu", ops::as_new_gelu_rule)
        .rewrite(&(), model)
    }
}
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
            .with_rule_for("detect-rms-norm", ops::as_rms_norm_rule)
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
            .with_rule_for("detect-rotate-half", ops::as_rotate_half_rule)
            .with_rule_for("detect-apply-rope", ops::as_apply_rope_rule)
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
        Rewriter::default().with_rule_for("detect-silu", ops::as_silu_rule).rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct ScaledMaskedSoftmaxTransform;

impl ModelTransform for ScaledMaskedSoftmaxTransform {
    fn name(&self) -> Cow<str> {
        "scaled-masked-softmax-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-scaled-masked-softmax", ops::as_scaled_masked_softmax_rule)
            .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct GeluTransform;

impl ModelTransform for GeluTransform {
    fn name(&self) -> Cow<str> {
        "gelu-fast-approx-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-gelu-approx", ops::as_gelu_approx_rule)
            .rewrite(&(), model)
    }
}

// TODO: This is why Transform shoudl be renamed to Remodel
#[derive(Debug, Default)]
pub struct TransformersTransform;

impl ModelTransform for TransformersTransform {
    fn name(&self) -> Cow<str> {
        "transformers-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-rms-norm", ops::as_rms_norm_rule)
            .with_rule_for("remove_rms_norm_cast", ops::remove_rms_norm_cast)
            .with_rule_for("detect-rotate-half", ops::as_rotate_half_rule)
            .with_rule_for("detect-apply-rope", ops::as_apply_rope_rule)
            .with_rule_for("detect-scaled-masked-softmax", ops::as_scaled_masked_softmax_rule)
            .with_rule_for("detect-silu", ops::as_silu_rule)
            .with_rule_for("detect-gelu-approx", ops::as_gelu_approx_rule)
            .rewrite(&(), model)
    }
}

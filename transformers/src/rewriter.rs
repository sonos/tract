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
            .with_rule_for("detect-rms-norm", ops::rms_norm_rule)
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
            .with_rule_for("detect-rotate-half", ops::rotate_half_rule)
            .with_rule_for("detect-apply-rope", ops::apply_rope_rule)
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
        Rewriter::default().with_rule_for("detect-silu", ops::silu_rule).rewrite(&(), model)
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
            .with_rule_for("detect-scaled-masked-softmax", ops::scaled_masked_softmax_rule)
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
            .with_rule_for("detect-gelu-approx", ops::gelu_approx_rule)
            .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct KeyValueCacheTransform;

impl ModelTransform for KeyValueCacheTransform {
    fn name(&self) -> Cow<str> {
        "gelu-fast-approx-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-kv-cache", ops::dynamic_kv_cache_rule)
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
            .with_rule_for("detect-kv-cache", ops::dynamic_kv_cache_rule)
            .with_rule_for("detect-rms-norm", ops::rms_norm_rule)
            .with_rule_for("detect-rotate-half", ops::rotate_half_rule)
            .with_rule_for("detect-apply-rope", ops::apply_rope_rule)
            .with_rule_for("detect-scaled-masked-softmax", ops::scaled_masked_softmax_rule)
            .with_rule_for("detect-silu", ops::silu_rule)
            .with_rule_for("detect-gelu-approx", ops::gelu_approx_rule)
            .rewrite(&(), model)
    }
}

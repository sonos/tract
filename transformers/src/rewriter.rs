use tract_nnef::internal::*;
use tract_nnef::tract_core::transform::ModelTransform;

use crate::ops;

#[derive(Debug, Default)]
pub struct ApplyRopeTransform;

impl ModelTransform for ApplyRopeTransform {
    fn name(&self) -> StaticName {
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
pub struct ScaledMaskedSoftmaxTransform;

impl ModelTransform for ScaledMaskedSoftmaxTransform {
    fn name(&self) -> StaticName {
        "scaled-masked-softmax-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-scaled-masked-softmax", ops::scaled_masked_softmax_rule)
            .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct KeyValueCacheTransform;

impl ModelTransform for KeyValueCacheTransform {
    fn name(&self) -> StaticName {
        "dynamic-kv-cache-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let inputs = model.inputs.clone();

        for input in inputs {
            ops::replace_kv_cache(model, input.node)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct SdpaFuseKvCacheBroadcastTransform;

impl ModelTransform for SdpaFuseKvCacheBroadcastTransform {
    fn name(&self) -> StaticName {
        "sdpa-fuse-kv-cache-broadcast-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-sdpa-kv-cache-broadcast", ops::fuse_kv_cache_broadcast_rule)
            .rewrite(&(), model)
    }
}

// TODO: This is why Transform should be renamed to Remodel
#[derive(Debug, Default)]
pub struct TransformersTransform;

impl ModelTransform for TransformersTransform {
    fn name(&self) -> StaticName {
        "transformers-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        KeyValueCacheTransform.transform(model)?;

        Rewriter::default()
            .with_rule_for("detect-rotate-half", ops::rotate_half_rule)
            .with_rule_for("detect-apply-rope", ops::apply_rope_rule)
            .with_rule_for("detect-scaled-masked-softmax", ops::scaled_masked_softmax_rule)
            .with_rule_for("detect-sdpa-kv-cache-broadcast", ops::fuse_kv_cache_broadcast_rule)
            .rewrite(&(), model)
    }
}

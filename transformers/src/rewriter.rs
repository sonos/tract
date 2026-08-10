use tract_nnef::internal::*;
use tract_nnef::tract_core::transform::ModelTransform;

use crate::ops;

/// Escape hatch for the rope detection rules (dyn-slice fold, identity-cast
/// fold, rotate-half both forms, apply-rope).
fn rope_detection_disabled() -> bool {
    std::env::var("TRACT_TRANSFORMERS_DISABLE_APPLY_ROPE").is_ok_and(|v| v == "1")
}

#[derive(Debug, Default)]
pub struct ApplyRopeTransform;

impl ModelTransform for ApplyRopeTransform {
    fn name(&self) -> StaticName {
        "detect_apply_rope".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        if rope_detection_disabled() {
            return Ok(());
        }
        // Exported rope subgraphs may compute their slice bounds from
        // shape_of chains; fold those to constants so the dyn-slice rule
        // below can turn them into the static Slice ops the patterns expect.
        model.prop_consts()?;
        Rewriter::default()
            .with_rule_for("fold-const-dyn-slice", ops::fold_const_dyn_slice_rule)
            .with_rule_for("fold-identity-cast", ops::fold_identity_cast_rule)
            .with_rule_for("detect-rotate-half", ops::rotate_half_rule)
            .with_rule_for("detect-rotate-half-concat-pair", ops::rotate_half_concat_pair_rule)
            .with_rule_for("detect-apply-rope", ops::apply_rope_rule)
            .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct ScaledMaskedSoftmaxTransform;

impl ModelTransform for ScaledMaskedSoftmaxTransform {
    fn name(&self) -> StaticName {
        "detect_scaled_masked_softmax".into()
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
        "detect_kv_cache".into()
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
        "detect_sdpa_kv_cache_broadcast".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-sdpa-kv-cache-broadcast", ops::fuse_kv_cache_broadcast_rule)
            .rewrite(&(), model)
    }
}

#[derive(Debug, Default)]
pub struct UnfoldKeyValueCacheTransform;

impl ModelTransform for UnfoldKeyValueCacheTransform {
    fn name(&self) -> StaticName {
        "unfold_kv_cache".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let kv_node_ids: Vec<usize> = model
            .nodes()
            .iter()
            .filter(|n| n.op_as::<ops::DynKeyValueCache>().is_some())
            .map(|n| n.id)
            .collect();
        for id in kv_node_ids {
            ops::unfold_kv_cache(model, id)?;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct DetectDiagGatherTransform;

impl ModelTransform for DetectDiagGatherTransform {
    fn name(&self) -> StaticName {
        "detect_diag_gather".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("detect-diag-gather", ops::diag_gather_rule)
            .rewrite(&(), model)
    }
}

// TODO: This is why Transform should be renamed to Remodel
#[derive(Debug, Default)]
pub struct TransformersTransform;

impl ModelTransform for TransformersTransform {
    fn name(&self) -> StaticName {
        "transformers_detect_all".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        KeyValueCacheTransform.transform(model)?;

        let mut rewriter = Rewriter::default();
        if !rope_detection_disabled() {
            // See ApplyRopeTransform: fold shape_of-derived slice bounds so
            // dyn-slice-based rope exports match the detection patterns.
            model.prop_consts()?;
            rewriter = rewriter
                .with_rule_for("fold-const-dyn-slice", ops::fold_const_dyn_slice_rule)
                .with_rule_for("fold-identity-cast", ops::fold_identity_cast_rule)
                .with_rule_for("detect-rotate-half", ops::rotate_half_rule)
                .with_rule_for(
                    "detect-rotate-half-concat-pair",
                    ops::rotate_half_concat_pair_rule,
                )
                .with_rule_for("detect-apply-rope", ops::apply_rope_rule);
        } else {
            rewriter = rewriter
                .with_rule_for("detect-rotate-half", ops::rotate_half_rule)
                .with_rule_for("detect-apply-rope", ops::apply_rope_rule);
        }
        rewriter
            .with_rule_for("detect-scaled-masked-softmax", ops::scaled_masked_softmax_rule)
            .with_rule_for("detect-sdpa-kv-cache-broadcast", ops::fuse_kv_cache_broadcast_rule)
            .with_rule_for("detect-diag-gather", ops::diag_gather_rule)
            .rewrite(&(), model)
    }
}

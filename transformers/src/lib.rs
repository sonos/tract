pub mod ops;
mod rewriter;
use std::collections::HashSet;

use rewriter::*;
use tract_nnef::internal::*;
use tract_nnef::tract_core::transform::ModelTransform;

pub fn get_transform(name: &str) -> Option<Box<dyn ModelTransform>> {
    match name {
        "detect-rms-norm" => Some(Box::new(RmsNormTransform)),
        "detect-apply-rope" => Some(Box::new(ApplyRopeTransform)),
        "detect-silu" => Some(Box::new(SiluTransform)),
        "detect-scaled-masked-softmax" => Some(Box::new(ScaledMaskedSoftmaxTransform)),
        "detect-gelu-approx" => Some(Box::new(GeluTransform)),
        "detect-kv-cache" => Some(Box::new(KeyValueCacheTransform)),
        "detect-sdpa-kv-cache-broadcast" => Some(Box::new(SdpaFuseKvCacheBroadcastTransform)),
        "transformers-detect-all" => Some(Box::new(TransformersTransform)),
        _ => None,
    }
}

pub fn register(registry: &mut Registry) {
    registry.transforms = Box::new(|s| Ok(get_transform(s)));

    ops::rms_norm::register(registry);
    ops::silu::register(registry);
    ops::gelu_approximate::register(registry);
    ops::apply_rope::register(registry);
    ops::scaled_masked_softmax::register(registry);
    ops::sdpa::register(registry);
}

pub trait WithTractTransformers {
    fn enable_tract_transformers(&mut self);
    fn with_tract_transformers(self) -> Self;
}

impl WithTractTransformers for tract_nnef::framework::Nnef {
    fn enable_tract_transformers(&mut self) {
        self.enable_tract_core();
        self.registries.push(tract_transformers_registry());
    }

    fn with_tract_transformers(mut self) -> Self {
        self.enable_tract_transformers();
        self
    }
}

pub fn tract_transformers_registry() -> Registry {
    let mut reg = Registry::new("tract_transformers")
        .with_doc("Extension `tract_transformers` extends NNEF with operators")
        .with_doc("for transformer networks.")
        .with_doc("")
        .with_doc("Add `extension tract_transformers` to `graph.nnef`");

    register(&mut reg);
    reg
}

pub fn figure_out_causal_llm_b_s_p(
    model: &TypedModel,
) -> TractResult<(Option<Symbol>, Option<Symbol>, Option<Symbol>)> {
    // expectations:
    // - one input is for tokens, so integer dt (i64 ?) and typically of shape S or 1,S, or B,S
    // - other inputs are kv cache, some kind of float. shape features both S and P, and B if B is present in tokens
    let token_input = model
        .inputs
        .iter()
        .position(|i| model.outlet_fact(*i).unwrap().datum_type.is_integer())
        .context("No token input found")?;
    let tokens_symbols = model.input_fact(token_input)?.shape.volume().symbols();
    let kv_symbols = if let Some(kv_input) =
        model.inputs.iter().position(|i| model.outlet_fact(*i).unwrap().datum_type.is_float())
    {
        model.input_fact(kv_input)?.shape.volume().symbols()
    } else {
        // Look for KVCache Op
        let dummy_session_state = TurnState::default();
        let mut symbols = HashSet::new();
        for node in &model.nodes {
            if let Some((_, fact)) =
                node.op.state(&dummy_session_state, 0)?.and_then(|state| state.init_tensor_fact())
            {
                symbols = fact.shape.volume().symbols();
                break;
            }
        }
        symbols
    };

    let b = tokens_symbols.intersection(&kv_symbols).cloned().collect::<HashSet<_>>();
    let s = tokens_symbols.difference(&b).cloned().collect::<HashSet<_>>();
    let p = kv_symbols.difference(&b).cloned().collect::<HashSet<_>>();
    Ok((b.into_iter().next(), s.into_iter().next(), p.into_iter().next()))
}

pub fn memory_arena_hints_for_causal_llm(model: &TypedModel) -> TractResult<SymbolValues> {
    let (b, s, p) = figure_out_causal_llm_b_s_p(model)?;
    let mut values = SymbolValues::default()
        .with(&s.context("Could not determine sequence_len (S)")?, 1024)
        .with(&p.context("Could not determine past_sequence_len (P)")?, 0);
    if let Some(b) = b {
        values = values.with(&b, 1);
    }
    Ok(values)
}

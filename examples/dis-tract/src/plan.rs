//! Memory-weighted layer planner: pick contiguous layer ranges so each shard's
//! weight bytes track the memory budget of the node that will hold it — the same
//! idea as EXO's ring memory-weighted partitioning.

use anyhow::{Result, bail};
use tract_core::prelude::*;

/// Per-layer weight-byte profile of a decoder LLM. Declutter renames nodes, so a
/// constant is attributed by the graph, not its name: to the layer of the earliest
/// block that consumes it (its consumers' cache-depth). Constants no block consumes
/// (embedding, lm_head, final norm) fold into the first/last bucket so the total is
/// conserved.
pub fn layer_weight_profile(model: &TypedModel, n_layers: usize) -> Vec<u64> {
    let depth = crate::llm::cache_depths(model).unwrap_or_default();
    let layer_of_node = |n: usize| depth.get(n).copied().flatten();
    let mut per_layer = vec![0u64; n_layers.max(1)];
    let (mut pre, mut post) = (0u64, 0u64);
    for node in model.nodes() {
        for out in &node.outputs {
            let Some(t) = out.fact.konst.as_ref() else { continue };
            let bytes = crate::partition::tensor_bytes(t) as u64;
            let layer = out.successors.iter().filter_map(|s| layer_of_node(s.node)).min();
            match layer {
                Some(l) if l < n_layers => per_layer[l] += bytes,
                Some(_) => post += bytes,
                None => pre += bytes,
            }
        }
    }
    per_layer[0] += pre;
    let last = per_layer.len() - 1;
    per_layer[last] += post;
    per_layer
}

/// Choose `budgets.len() - 1` cut layers (each = the first layer of the next
/// stage) so cumulative weight per stage tracks the cumulative budget fraction,
/// in the given node order. Cuts are strictly increasing within `1..n_layers`.
pub fn memory_weighted_cuts(profile: &[u64], budgets: &[u64]) -> Result<Vec<usize>> {
    let n_stages = budgets.len();
    if n_stages < 2 {
        return Ok(vec![]);
    }
    let n_layers = profile.len();
    if n_layers < n_stages {
        bail!("{n_layers} layers cannot be split across {n_stages} stages");
    }
    let total_w: f64 = profile.iter().sum::<u64>() as f64;
    let total_b: f64 = budgets.iter().sum::<u64>() as f64;
    if total_b == 0.0 {
        bail!("all node memory budgets are zero");
    }

    let mut cuts = Vec::with_capacity(n_stages - 1);
    let mut cum_b = 0.0;
    let mut cum_w = 0.0;
    let mut layer = 0usize;
    for (k, b) in budgets[..n_stages - 1].iter().enumerate() {
        cum_b += *b as f64;
        let target = total_w * (cum_b / total_b);
        while layer < n_layers && cum_w < target {
            cum_w += profile[layer] as f64;
            layer += 1;
        }
        // Keep cuts strictly increasing and leave room for the remaining stages.
        let lo = cuts.last().map(|&c| c + 1).unwrap_or(1);
        let hi = n_layers - (n_stages - 1 - k);
        cuts.push(layer.clamp(lo, hi));
        layer = *cuts.last().unwrap();
    }
    Ok(cuts)
}

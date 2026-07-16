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

const MIB: u64 = 1024 * 1024;

/// Weight bytes each stage owns, given the cut layers. Stage `s` owns every layer
/// `l` with `cuts.iter().filter(|c| **c <= l).count() == s`.
pub fn stage_weights(profile: &[u64], cuts: &[usize]) -> Vec<u64> {
    let owner = |l: usize| cuts.iter().filter(|&&c| c <= l).count();
    let mut w = vec![0u64; cuts.len() + 1];
    for (l, bytes) in profile.iter().enumerate() {
        w[owner(l)] += bytes;
    }
    w
}

/// Reject a split that hands a node more weight than it said it could hold.
///
/// Weights are a **lower bound** on what a stage costs: its KV cache, activations
/// and runtime are on top, and measured 1.2-1.6x of weights on an 8B model. So this
/// catches a node that cannot possibly fit its shard, not one that merely might not.
fn check_fit(profile: &[u64], cuts: &[usize], budgets: &[u64]) -> Result<()> {
    for (s, (need, budget)) in stage_weights(profile, cuts).iter().zip(budgets).enumerate() {
        if need > budget {
            bail!(
                "stage {s} would hold {} MiB of weights but its node advertised {} MiB \
                 (weights alone, before KV and runtime): raise its --mem-mb if the node \
                 really has the memory, add nodes, or use a smaller model",
                need / MIB,
                budget / MIB
            );
        }
    }
    Ok(())
}

/// Choose `budgets.len() - 1` cut layers (each = the first layer of the next
/// stage) so cumulative weight per stage tracks the cumulative budget fraction,
/// in the given node order. Cuts are strictly increasing within `1..n_layers`.
///
/// The split is proportional, so a node's share can exceed its budget when the
/// cluster as a whole is short of memory; [`check_fit`] turns that into a planning
/// error rather than a load-time failure on one unlucky worker.
pub fn memory_weighted_cuts(profile: &[u64], budgets: &[u64]) -> Result<Vec<usize>> {
    let n_stages = budgets.len();
    let n_layers = profile.len();
    if n_stages < 2 {
        check_fit(profile, &[], budgets)?;
        return Ok(vec![]);
    }
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
    check_fit(profile, &cuts, budgets)?;
    Ok(cuts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * MIB;

    #[test]
    fn weights_follow_the_cuts() {
        // 4 layers of 1 GB, cut at 2 -> two stages of 2 GB.
        assert_eq!(stage_weights(&[GB, GB, GB, GB], &[2]), vec![2 * GB, 2 * GB]);
        // No cuts: one stage owns everything.
        assert_eq!(stage_weights(&[GB, GB], &[]), vec![2 * GB]);
    }

    #[test]
    fn equal_budgets_split_evenly() {
        let cuts = memory_weighted_cuts(&[GB, GB, GB, GB], &[4 * GB, 4 * GB]).unwrap();
        assert_eq!(stage_weights(&[GB, GB, GB, GB], &cuts), vec![2 * GB, 2 * GB]);
    }

    #[test]
    fn a_bigger_budget_takes_more_layers() {
        let profile = [GB, GB, GB, GB];
        let cuts = memory_weighted_cuts(&profile, &[3 * GB, GB]).unwrap();
        let w = stage_weights(&profile, &cuts);
        assert!(w[0] > w[1], "stage 0 advertised 3x the budget but got {w:?}");
    }

    #[test]
    fn a_shard_over_its_node_budget_fails_at_plan_time() {
        // 8 GB of weights across two 1 GB nodes: proportional, and neither fits.
        let err = memory_weighted_cuts(&[4 * GB, 4 * GB], &[GB, GB]).unwrap_err().to_string();
        assert!(err.contains("advertised"), "unhelpful error: {err}");
        assert!(err.contains("4096 MiB"), "should name what it needs: {err}");
    }

    #[test]
    fn a_single_node_is_checked_too() {
        // The one-stage path returns before any split, but must still refuse a model
        // that cannot fit the only node.
        assert!(memory_weighted_cuts(&[4 * GB], &[GB]).is_err());
        assert!(memory_weighted_cuts(&[GB], &[4 * GB]).is_ok());
    }
}

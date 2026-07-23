//! Compare what the CPU optimizer produces for the whole model versus an
//! extracted sub-model, around the last transformer block.
//!
//!   distract-dump <model.nnef.tgz>

use tract_core::model::extract_subgraph;
use tract_core::prelude::*;
use tract_distributed::llm::{cache_depths, load_model};

fn histogram(model: &TypedModel, label: &str) -> anyhow::Result<()> {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for n in model.nodes() {
        *counts.entry(n.op().name().to_string()).or_default() += 1;
    }
    println!("\n=== {label}: {} nodes ===", model.nodes().len());
    for (op, c) in counts.iter().filter(|(_, c)| **c > 0) {
        println!("  {c:>4}  {op}");
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).expect("model");
    let (full, _n) = load_model(&path)?;
    let depth = cache_depths(&full)?;
    let n_layers = full
        .input_outlets()?
        .iter()
        .filter(|o| full.node(o.node).name.contains("cache_key"))
        .count();
    let last = n_layers - 1;

    let opt_full = full.clone().into_optimized()?;
    histogram(&opt_full, "full model, optimized")?;

    let ins = full.input_outlets()?.to_vec();
    let watched: Vec<OutletId> = full
        .eval_order()?
        .iter()
        .copied()
        .filter(|&n| depth[n] == Some(last))
        .filter(|&n| !full.node(n).inputs.is_empty())
        .map(|n| OutletId::from((n, 0)))
        .collect();
    let mut outs = vec![full.output_outlets()?[0]];
    outs.extend(watched);
    let opt_sub = extract_subgraph(&full, &ins, &outs)?.into_optimized()?;
    histogram(&opt_sub, "extracted sub-model, optimized")?;

    for (label, m) in [("FULL", &opt_full), ("SUB", &opt_sub)] {
        println!("\n--- {label}: attention ops of the last block ---");
        for n in m.nodes() {
            let op = n.op().name().to_string();
            if !n.name.contains(&format!("_{last}_selfAttn")) {
                continue;
            }
            if !(op.contains("Sdpa") || op.contains("MatMul") || op.contains("Softmax")) {
                continue;
            }
            let its: Vec<String> = n
                .inputs
                .iter()
                .map(|i| format!("{:?}", m.outlet_fact(*i).map(|f| f.datum_type)))
                .collect();
            println!(
                "  {op:<24} out={:?}  in={}",
                m.outlet_fact(OutletId::from((n.id, 0)))?.datum_type,
                its.join(",")
            );
        }
    }
    Ok(())
}

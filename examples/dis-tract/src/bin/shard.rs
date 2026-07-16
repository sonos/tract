//! Validate the per-shard graph pruner on a real NNEF graph.
//!
//!   `distract-shard <graph.nnef> <n_layers> <start> <end>`

use std::collections::BTreeSet;
use tract_distributed::shard_graph::{label_layer, plan_shard};
use tract_nnef::ast::parse::parse_document;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("graph.nnef");
    let n_layers: usize = std::env::args().nth(2).expect("n_layers").parse()?;
    let start: usize = std::env::args().nth(3).expect("start").parse()?;
    let end: usize = std::env::args().nth(4).expect("end").parse()?;

    let text = std::fs::read_to_string(&path)?;
    let doc = parse_document(&text)?;
    let total = doc.graph_def.body.len();

    let plan = plan_shard(&doc, start, end, n_layers)?;

    // Layer distribution of the weights this shard would load.
    let mut layers: BTreeSet<usize> = BTreeSet::new();
    let mut globals = 0usize;
    for l in &plan.weight_labels {
        match label_layer(l) {
            Some(n) => {
                layers.insert(n);
            }
            None => globals += 1,
        }
    }

    println!("shard [{start},{end}) of {n_layers}  ({total} total assignments)");
    println!("  kept assignments : {}", plan.keep.len());
    println!(
        "  weights          : {} ({} global, {} layer-scoped)",
        plan.weight_labels.len(),
        globals,
        plan.weight_labels.len() - globals
    );
    println!("  layers touched   : {:?}", layers);
    println!("  inputs  ({:2})     : {}", plan.inputs.len(), preview(&plan.inputs));
    println!("  outputs ({:2})     : {}", plan.outputs.len(), preview(&plan.outputs));

    // Sanity: no weight outside [start,end) should be referenced.
    let leak: Vec<_> = layers.iter().filter(|n| **n < start || **n >= end).collect();
    if leak.is_empty() {
        println!("  OK: no out-of-range layer weights referenced");
    } else {
        println!("  LEAK: references weights of layers {leak:?}");
    }
    // Show a few sample globals so we can eyeball the embedding.
    let sample_globals: Vec<_> =
        plan.weight_labels.iter().filter(|l| label_layer(l).is_none()).take(6).collect();
    println!("  sample globals   : {sample_globals:?}");
    Ok(())
}

fn preview(v: &[String]) -> String {
    if v.len() <= 6 {
        v.join(", ")
    } else {
        format!("{}, … (+{} more)", v[..6].join(", "), v.len() - 6)
    }
}

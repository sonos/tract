//! Dev tool: dump the neighborhood of every Softmax node in an NNEF export,
//! to design attention fuse rules against the real exported pattern.
//!
//!   cargo run --release -p tract-transformers --example dump_attn_subgraph -- <model.nnef.tgz> [max_softmax]

use std::collections::HashSet;

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::nn::Softmax;
use tract_transformers::WithTractTransformers;

fn describe(model: &TypedModel, id: usize) -> String {
    let n = model.node(id);
    let facts: Vec<String> = n
        .outputs
        .iter()
        .map(|o| format!("{:?}", o.fact))
        .collect();
    let ins: Vec<String> = n.inputs.iter().map(|i| format!("{}/{}", i.node, i.slot)).collect();
    format!(
        "#{id} {} [{}] inputs=[{}] -> {}",
        n.op.name(),
        n.name,
        ins.join(", "),
        facts.join(" ; ")
    )
}

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: dump_attn_subgraph <model.nnef.tgz> [max]");
    let max: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);

    let nnef = tract_nnef::nnef().with_tract_transformers();
    let model = nnef.model_for_path(&path)?.into_decluttered()?;
    eprintln!("model loaded: {} nodes", model.nodes().len());

    let softmaxes: Vec<usize> = model
        .nodes()
        .iter()
        .filter(|n| n.op_is::<Softmax>())
        .map(|n| n.id)
        .collect();
    eprintln!("{} Softmax nodes", softmaxes.len());

    for (i, &sm) in softmaxes.iter().enumerate().take(max) {
        println!("=========== softmax #{i} (node {sm}) ===========");
        // Upstream BFS depth 12
        let mut seen = HashSet::new();
        let mut frontier = vec![(sm, 0usize)];
        let mut lines = vec![];
        while let Some((id, depth)) = frontier.pop() {
            if !seen.insert(id) || depth > 14 {
                continue;
            }
            lines.push((id, depth));
            for input in &model.node(id).inputs {
                frontier.push((input.node, depth + 1));
            }
        }
        lines.sort();
        println!("--- upstream (depth<=14) ---");
        for (id, depth) in &lines {
            println!("{}{}", "  ".repeat(*depth), describe(&model, *id));
        }
        // Downstream BFS depth 10
        let mut seen = HashSet::new();
        let mut frontier = vec![(sm, 0usize)];
        println!("--- downstream (depth<=10) ---");
        let mut down = vec![];
        while let Some((id, depth)) = frontier.pop() {
            if !seen.insert(id) || depth > 10 {
                continue;
            }
            down.push((id, depth));
            for out in &model.node(id).outputs {
                for succ in &out.successors {
                    frontier.push((succ.node, depth + 1));
                }
            }
        }
        down.sort();
        for (id, depth) in &down {
            println!("{}{}", "  ".repeat(*depth), describe(&model, *id));
        }
    }
    // Also: model outputs
    println!("--- model outputs ---");
    for (i, o) in model.outputs.iter().enumerate() {
        println!("out[{i}] = node {} slot {} ({})", o.node, o.slot, model.node(o.node).name);
    }
    Ok(())
}

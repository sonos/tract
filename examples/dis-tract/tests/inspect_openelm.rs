//! Read-only structural probe of a real LLM (run with DISTRACT_MODEL set).
//! `DISTRACT_MODEL=/path/model.nnef.tgz cargo test -p tract-distributed \
//!    --test inspect_openelm -- --ignored --nocapture`

use std::collections::BTreeMap;

use anyhow::Result;
use tract_core::prelude::*;
use tract_distributed::codec;

#[test]
#[ignore]
fn inspect() -> Result<()> {
    let path = std::env::var("DISTRACT_MODEL").expect("set DISTRACT_MODEL");
    let model = codec::nnef().model_for_path(&path)?;
    println!("== {} nodes ==", model.nodes().len());

    for (i, o) in model.input_outlets()?.iter().enumerate() {
        println!("input  {i}: {:40} {:?}", model.node(o.node).name, model.outlet_fact(*o)?);
    }
    for (i, o) in model.output_outlets()?.iter().enumerate() {
        println!("output {i}: {:40} {:?}", model.node(o.node).name, model.outlet_fact(*o)?);
    }

    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for n in model.nodes() {
        *hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("== op histogram ==");
    for (k, v) in &hist {
        println!("  {v:5}  {k}");
    }

    // Try folding external KV I/O back to internal state.
    let mut folded = model.clone();
    for t in ["transformers_detect_all", "detect_kv_cache"] {
        if let Some(tr) = tract_core::transform::get_transform(t)? {
            match tr.transform(&mut folded) {
                Ok(()) => println!("applied transform {t}"),
                Err(e) => println!("transform {t} FAILED: {e}"),
            }
        } else {
            println!("no transform named {t}");
        }
    }
    println!(
        "after fold: {} inputs, {} outputs, {} nodes",
        folded.input_outlets()?.len(),
        folded.output_outlets()?.len(),
        folded.nodes().len()
    );
    for (i, o) in folded.input_outlets()?.iter().enumerate() {
        println!("  fold input  {i}: {}", folded.node(o.node).name);
    }
    for (i, o) in folded.output_outlets()?.iter().enumerate() {
        println!("  fold output {i}: {}", folded.node(o.node).name);
    }

    // Trace the residual stream: rank-3 Add nodes [1,S,D] with a single dominant
    // model dim D. These are the block boundaries we can cut on.
    println!("== rank-3 Add nodes (residual-stream candidates) ==");
    let mut count = 0;
    for n in model.nodes() {
        if n.op.name() != "Add" {
            continue;
        }
        let fact = &n.outputs[0].fact;
        let shape = fact.shape.to_tvec();
        if shape.len() == 3 {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            println!("  {:50} [{}]", n.name, dims.join(","));
            count += 1;
        }
    }
    println!("total rank-3 Add nodes: {count}");
    Ok(())
}

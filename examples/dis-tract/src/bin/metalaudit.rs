//! Compare which ops land on the GPU for the whole model versus a shard, by
//! replicating the Metal runtime's own pipeline (`MetalTransform` then optimize)
//! and counting device vs host ops. A shard that keeps ops on the host is paying
//! per-op fallback the whole model avoids.
//!
//!   distract-metalaudit <model.nnef.tgz> <n_layers>

use std::collections::BTreeMap;

use anyhow::Result;
use tract_core::prelude::*;
use tract_core::transform::ModelTransform;
use tract_distributed::llm::load_model;
use tract_distributed::shard_graph::load_shard;

fn audit(label: &str, mut model: TypedModel) -> Result<()> {
    tract_metal::MetalTransform::default().transform(&mut model)?;
    let model = model.into_optimized()?;
    let mut counts: BTreeMap<String, usize> = Default::default();
    for n in model.nodes() {
        *counts.entry(n.op().name().to_string()).or_default() += 1;
    }
    let is_dev =
        |k: &str| k.starts_with("Metal") || k.starts_with("Device") || k.starts_with("Gpu");
    let dev: usize = counts.iter().filter(|(k, _)| is_dev(k)).map(|(_, c)| *c).sum();
    let host: usize = counts.iter().filter(|(k, _)| !is_dev(k)).map(|(_, c)| *c).sum();
    println!("\n[{label}] {} nodes: {dev} device, {host} host", model.nodes().len());
    let mut v: Vec<_> = counts.iter().filter(|(k, _)| !is_dev(k)).collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (k, c) in v.iter().take(10) {
        println!("    host {c:>5}  {k}");
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let path = std::env::args().nth(1).expect("model.nnef.tgz");
    let n_layers: usize = std::env::args().nth(2).expect("n_layers").parse()?;

    let (full, _) = load_model(&path)?;
    audit("load_model full", full)?;

    let (shard, _) = load_shard(&path, 0, n_layers, n_layers)?;
    audit("load_shard 0..n", shard)?;
    Ok(())
}

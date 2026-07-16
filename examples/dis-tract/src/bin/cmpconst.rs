//! Compare how load_model vs load_shard represent block-quant weight consts, to
//! find why Metal's tensor_to_device recognizes one but not the other.
//!
//!   distract-cmpconst <model.nnef.tgz> <n_layers>

use tract_core::prelude::*;
use tract_core::tract_linalg::block_quant::{BlockQuantStorage, Q4_0};
use tract_distributed::llm::load_model;
use tract_distributed::shard_graph::load_shard;

fn probe(label: &str, model: &TypedModel) {
    // The metal panic hits consts that are non-plain but NOT q40-recognized.
    let mut bad = vec![];
    for n in model.nodes() {
        let Some(k) = n.op_as::<tract_core::ops::konst::Const>() else { continue };
        let t = k.val();
        let recognized =
            t.storage_as::<BlockQuantStorage>().map(|b| b.format().dyn_eq(&Q4_0)).unwrap_or(false);
        if !t.is_plain() && !recognized {
            bad.push(format!("{} (dt={:?} shape={:?})", n.name, t.datum_type(), t.shape()));
        }
    }
    println!("[{label}] non-plain non-q40 consts (metal would panic on these): {}", bad.len());
    for b in bad.iter().take(6) {
        println!("        {b}");
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).expect("model");
    let n_layers: usize = std::env::args().nth(2).expect("n_layers").parse()?;

    let cut = n_layers / 2;
    let (full, _n) = load_model(&path)?;
    probe("load_model full", &full);
    drop(full);

    let (a, _) = load_shard(&path, 0, cut, n_layers)?;
    probe(&format!("shard [0,{cut})"), &a);
    drop(a);
    let (b, _) = load_shard(&path, cut, n_layers, n_layers)?;
    probe(&format!("shard [{cut},{n_layers})"), &b);
    // also the optimized forms, since metal transforms after optimize in one path
    let (b2, _) = load_shard(&path, cut, n_layers, n_layers)?;
    probe(&format!("shard [{cut},{n_layers}) OPTIMIZED"), &b2.into_optimized()?);

    if std::env::var("METAL").is_ok() {
        println!("=== attempting metal prepare of shard [{cut},{n_layers}) ===");
        let (b3, _) = load_shard(&path, cut, n_layers, n_layers)?;
        let rt = tract_core::runtime::runtime_for_name("metal")?
            .ok_or_else(|| anyhow::anyhow!("no metal runtime"))?;
        let _ = rt.prepare(b3)?;
        println!("metal prepare OK");
    }
    Ok(())
}

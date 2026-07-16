//! Bisect where a split's decode cost comes from: time decode steps for the whole
//! model, a full-range shard, and a 2-shard chain — all in one process on one
//! backend, so the only variable is how the model was built.
//!
//!   distract-shardbench <model.nnef.tgz> [backend] [turns]

use std::time::Instant;

use anyhow::Result;
use tract_core::prelude::*;
use tract_distributed::llm::{StageState, full_io_roles, load_model};
use tract_distributed::shard_graph::{load_shard, shard_io_roles};

fn ids(toks: &[i64]) -> Tensor {
    Tensor::from_shape(&[1, toks.len()], toks).unwrap()
}

fn argmax(t: &Tensor) -> Result<i64> {
    let f = t.cast_to::<f32>()?;
    let v = f.view();
    let s = v.as_slice::<f32>()?;
    let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, &x) in s.iter().enumerate() {
        if !x.is_nan() && x > bv {
            (best, bv) = (i, x);
        }
    }
    Ok(best as i64)
}

/// Prefill then decode `turns` tokens through the chain, reporting the decode-step
/// latency (prefill excluded) and the tokens, so a config's speed and its parity
/// against the others are visible together.
fn time_chain(label: &str, chain: &mut [StageState], turns: usize) -> Result<Vec<i64>> {
    let prompt = vec![9707i64, 11, 1246, 525, 498, 304, 264, 3283];
    let mut tok = 0i64;
    let mut toks = vec![];
    let mut decode_ms: Vec<f64> = vec![];
    for turn in 0..turns {
        let t0 = Instant::now();
        let mut wire: TVec<Tensor> = tvec!(if turn == 0 { ids(&prompt) } else { ids(&[tok]) });
        for st in chain.iter_mut() {
            wire = st.step(wire)?;
        }
        tok = argmax(&wire[0])?;
        toks.push(tok);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if turn > 0 {
            decode_ms.push(ms);
        }
    }
    let n = decode_ms.len().max(1) as f64;
    let mean = decode_ms.iter().sum::<f64>() / n;
    let min = decode_ms.iter().cloned().fold(f64::MAX, f64::min);
    println!(
        "{label:26} decode mean {mean:6.1} ms   min {min:6.1} ms   {:5.1} tok/s",
        1000.0 / mean
    );
    Ok(toks)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let path = std::env::args().nth(1).expect("model.nnef.tgz");
    let backend = std::env::args().nth(2).unwrap_or_else(|| "metal".into());
    let turns: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    let (full, n_regular) = load_model(&path)?;
    let n_layers = full
        .input_outlets()?
        .iter()
        .filter(|o| full.node(o.node).name.contains("cache_key"))
        .count();
    println!("model {n_layers} layers, backend {backend}, {turns} turns\n");

    let a = {
        let (fin, fout) = full_io_roles(&full, n_regular)?;
        let mut c = [StageState::new(full.clone(), &backend, fin, fout)?];
        time_chain("A whole (load_model)", &mut c, turns)?
    };
    drop(full);

    let b = {
        let (m, _) = load_shard(&path, 0, n_layers, n_layers)?;
        let (i, o) = shard_io_roles(&m.clone().into_optimized()?)?;
        let mut c = [StageState::new(m, &backend, i, o)?];
        time_chain("B shard 0..n (load_shard)", &mut c, turns)?
    };

    let cut = n_layers / 2;
    let c = {
        let mut c = vec![];
        for (s, e) in [(0, cut), (cut, n_layers)] {
            let (m, _) = load_shard(&path, s, e, n_layers)?;
            let (i, o) = shard_io_roles(&m.clone().into_optimized()?)?;
            c.push(StageState::new(m, &backend, i, o)?);
        }
        time_chain("C 2-shard chain", &mut c, turns)?
    };

    println!("\ntokens A {a:?}\ntokens B {b:?}\ntokens C {c:?}");
    println!(
        "parity: B vs A {}   C vs A {}",
        if b == a { "OK" } else { "MISMATCH" },
        if c == a { "OK" } else { "MISMATCH" }
    );
    Ok(())
}

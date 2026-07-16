//! Verify the per-block modular split: chain one module per transformer block
//! and check token-for-token parity against the whole model.
//!
//!   distract-probe <model.nnef.tgz> [backend]

use tract_core::prelude::*;
use tract_distributed::llm::{StageState, full_io_roles, load_model, split_blocks};

/// Returns (argmax, nan_count).
fn argmax(t: &Tensor) -> (i64, usize) {
    let f = t.cast_to::<f32>().unwrap();
    let v = f.view();
    let s = v.as_slice::<f32>().unwrap();
    let (mut best, mut bv, mut nans) = (0usize, f32::NEG_INFINITY, 0usize);
    for (i, &x) in s.iter().enumerate() {
        if x.is_nan() {
            nans += 1;
        } else if x > bv {
            (best, bv) = (i, x);
        }
    }
    (best as i64, nans)
}

fn ids(toks: &[i64]) -> Tensor {
    Tensor::from_shape(&[1, toks.len()], toks).unwrap()
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).expect("model path");
    let backend = std::env::args().nth(2).unwrap_or_else(|| "cpu".into());

    // Wire codec sanity: Qwen's frontier sends a bool mask + f16 activations
    // between stages, so both must survive the NNEF tensor round-trip.
    for (label, t) in [
        ("bool", Tensor::from_shape(&[2, 3], &[true, false, true, true, false, false])?),
        (
            "f16",
            Tensor::from_shape(&[2, 2], &[1.5f32, -2.25, 0.0, 7.5])?.cast_to::<f16>()?.into_owned(),
        ),
    ] {
        let back = tract_distributed::codec::tensor_from_bytes(
            &tract_distributed::codec::tensor_to_bytes(&t)?,
        )?;
        println!("codec {label} roundtrip: {}", if back == t { "OK" } else { "MISMATCH" });
    }

    let (full, n_regular) = load_model(&path)?;
    let n_layers = full
        .input_outlets()?
        .iter()
        .filter(|o| full.node(o.node).name.contains("cache_key"))
        .count();
    println!("model: {} nodes, {n_layers} blocks", full.nodes().len());

    // one module per block (reuses partition_stages)
    let blocks = split_blocks(&full, n_layers, n_regular)?;
    println!("split into {} block modules", blocks.len());

    // reference: the whole model through the same StageState machinery
    let (fin, fout) = full_io_roles(&full, n_regular)?;
    let mut reference = StageState::new(full.clone(), &backend, fin, fout)?;

    // one StageState per block module (skippable: the chain is slow to prepare on CPU)
    let ref_only = std::env::var("DISTRACT_REF_ONLY").is_ok();
    let mut chain: Vec<StageState> = vec![];
    if !ref_only {
        for b in blocks {
            chain.push(StageState::new(b.model, &backend, b.inputs, b.outputs)?);
        }
    }

    let prompt: Vec<i64> = match std::env::args().nth(3) {
        Some(s) => s.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        None => vec![9, 3, 128, 42],
    };
    let turns: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(6);
    println!("prompt {} tokens, {turns} turns", prompt.len());
    let (mut ref_tok, mut chain_tok) = (0i64, 0i64);
    let (mut matches, mut total) = (0usize, 0usize);
    for turn in 0..turns {
        let ref_in = if turn == 0 { ids(&prompt) } else { ids(&[ref_tok]) };
        let (rt, rnan) = argmax(&reference.step(tvec!(ref_in))?[0]);
        ref_tok = rt;

        if ref_only {
            println!("turn {turn}: ref={ref_tok}(nan {rnan})");
            continue;
        }
        // run the prompt/token through every block module in sequence
        let mut wire: TVec<Tensor> =
            tvec!(if turn == 0 { ids(&prompt) } else { ids(&[chain_tok]) });
        for (bi, st) in chain.iter_mut().enumerate() {
            wire = st.step(wire)?;
            if std::env::var("DISTRACT_TRACE").is_ok() {
                // residual is wire[0]; watch its magnitude vs the f16 ceiling (65504)
                let f = wire[0].cast_to::<f32>()?;
                let v = f.view();
                let s = v.as_slice::<f32>()?;
                let mx = s.iter().filter(|x| x.is_finite()).fold(0f32, |a, b| a.max(b.abs()));
                let bad = s.iter().filter(|x| !x.is_finite()).count();
                println!("    block {bi:2}: max|res| {mx:>10.1}  non-finite {bad}");
            }
        }
        let (ct, cnan) = argmax(&wire[0]);
        chain_tok = ct;

        total += 1;
        matches += (ref_tok == chain_tok) as usize;
        println!(
            "turn {turn}: ref={ref_tok}(nan {rnan}) blocks={chain_tok}(nan {cnan}) {}",
            if ref_tok == chain_tok { "OK" } else { "MISMATCH" }
        );
    }
    println!("per-block split parity: {matches}/{total} vs whole model");
    anyhow::ensure!(matches == total, "per-block split did not match the whole model");
    Ok(())
}

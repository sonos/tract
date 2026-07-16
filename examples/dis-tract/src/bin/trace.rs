//! Find the FIRST op that produces a non-finite value.
//!
//! Makes every node of the failing region an output of an extracted sub-model,
//! runs one prefill, and reports the first node (in eval order) whose output is
//! not finite — i.e. exactly where inf/NaN is born.
//!
//!   `distract-trace <model.nnef.tgz> <backend> <comma,separated,prompt,ids>`

use tract_core::model::extract_subgraph;
use tract_core::prelude::*;
use tract_distributed::llm::{cache_depths, load_model};
use tract_distributed::runner::StageRunner;

fn stats(t: &Tensor) -> Option<(f32, usize)> {
    let f = t.cast_to::<f32>().ok()?;
    let v = f.view();
    let s = v.as_slice::<f32>().ok()?;
    let bad = s.iter().filter(|x| !x.is_finite()).count();
    let mx = s.iter().filter(|x| x.is_finite()).fold(0f32, |a, b| a.max(b.abs()));
    Some((mx, bad))
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).expect("model");
    let backend = std::env::args().nth(2).unwrap_or_else(|| "metal".into());
    let prompt: Vec<i64> = std::env::args()
        .nth(3)
        .expect("prompt ids")
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();

    let (full, _n_regular) = load_model(&path)?;
    let depth = cache_depths(&full)?;
    let n_layers = full
        .input_outlets()?
        .iter()
        .filter(|o| full.node(o.node).name.contains("cache_key"))
        .count();
    let last = n_layers - 1;
    let min_depth: usize = std::env::var("DISTRACT_MIN_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(last.saturating_sub(1));

    // model inputs: token + every cache (seeded empty)
    let ins: Vec<OutletId> = full.input_outlets()?.to_vec();

    // watch every node of the last block + epilogue, in eval order
    let order = full.eval_order()?;
    let watched: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&n| depth[n].is_some_and(|d| d + 1 > min_depth))
        .filter(|&n| !full.node(n).inputs.is_empty())
        .collect();
    println!("watching {} nodes at depth >= {min_depth}", watched.len());

    // Keep the model's own outputs: dropping them changes what the optimizer fuses,
    // which can hide the very bug we are chasing.
    let logits = full.output_outlets()?[0];
    let mut outs: Vec<OutletId> = vec![logits];
    outs.extend(watched.iter().map(|&n| OutletId::from((n, 0))));
    let probe = extract_subgraph(&full, &ins, &outs)?;
    let mut runner = StageRunner::load(probe, &backend)?;

    // feed the prompt + empty caches
    let mut inputs: TVec<TValue> =
        tvec!(Tensor::from_shape(&[1, prompt.len()], &prompt)?.into_tvalue());
    for o in full.input_outlets()?.iter().skip(1) {
        let f = full.outlet_fact(*o)?;
        let shape: Vec<usize> = f
            .shape
            .iter()
            .enumerate()
            .map(|(ax, d)| d.to_i64().map(|v| v as usize).unwrap_or(if ax == 0 { 1 } else { 0 }))
            .collect();
        inputs.push(Tensor::zero_dt(f.datum_type, &shape)?.into_tvalue());
    }

    let all = runner.run(inputs)?;
    if let Some((mx, bad)) = stats(&all[0].clone().into_tensor()) {
        println!("logits: max|x|={mx:.1} non_finite={bad}");
    }
    let produced = &all[1..];
    let mut first = true;
    for (i, &n) in watched.iter().enumerate() {
        let node = full.node(n);
        let Some((mx, bad)) = stats(&produced[i].clone().into_tensor()) else { continue };
        if bad > 0 && first {
            println!("\n>>> FIRST NON-FINITE <<<");
            println!("    node   : {}", node.name);
            println!("    op     : {}", node.op().name());
            println!("    dt     : {:?}", full.outlet_fact(OutletId::from((n, 0)))?.datum_type);
            println!("    inputs :");
            for inp in &node.inputs {
                let pn = full.node(inp.node);
                let pf = full.outlet_fact(*inp)?;
                let pi = watched.iter().position(|&w| w == inp.node);
                let s = pi.and_then(|k| stats(&produced[k].clone().into_tensor()));
                println!(
                    "      {} [{}] dt={:?} {}",
                    pn.name,
                    pn.op().name(),
                    pf.datum_type,
                    match s {
                        Some((m, b)) => format!("max|x|={m:.1} non_finite={b}"),
                        None => "(not watched / const)".into(),
                    }
                );
            }
            first = false;

            // For an SDPA, recompute Q.K^T in f32 from the captured inputs: if the true
            // scores exceed f16 max (65504) the kernel cannot hold them in half.
            if node.op().name().contains("SDPA") && node.inputs.len() >= 3 {
                let get = |k: usize| -> Option<Tensor> {
                    let inp = node.inputs[k];
                    let pi = watched.iter().position(|&w| w == inp.node)?;
                    produced[pi].clone().into_tensor().cast_to::<f32>().ok().map(|c| c.into_owned())
                };
                if let (Some(q), Some(kk)) = (get(0), get(1)) {
                    let (qs, ks) = (q.shape().to_vec(), kk.shape().to_vec());
                    println!("    Q shape {qs:?}  K shape {ks:?}");
                    let (qh, kh, d) = (qs[1], ks[1], *qs.last().unwrap());
                    let (rq, rk) = (qs[2], ks[2]);
                    let (qview, kview) = (q.view(), kk.view());
                    let qv = qview.as_slice::<f32>()?;
                    let kv = kview.as_slice::<f32>()?;
                    let mut worst = 0f32;
                    for h in 0..qh {
                        let hk = h / (qh / kh);
                        for i in 0..rq {
                            for j in 0..rk {
                                let qo = (h * rq + i) * d;
                                let ko = (hk * rk + j) * d;
                                let dot: f32 = (0..d).map(|t| qv[qo + t] * kv[ko + t]).sum::<f32>();
                                worst = worst.max(dot.abs());
                            }
                        }
                    }
                    let scale = 1.0 / (d as f32).sqrt();
                    println!(
                        "    true max|Q.K^T| (f32) = {worst:.0}   scaled = {:.0}",
                        worst * scale
                    );
                    println!(
                        "    f16 max = 65504  =>  raw scores {} f16",
                        if worst > 65504.0 { "OVERFLOW" } else { "fit in" }
                    );
                }
            }
        }
        if bad > 0 || (std::env::var("DISTRACT_VERBOSE").is_ok() && mx > 1000.0) {
            println!(
                "  {:>4}: {:<62} {:<14} max {:>9.1} nf {bad}",
                i,
                node.name,
                node.op().name(),
                mx
            );
        }
    }
    if first {
        println!("no non-finite values in the watched region");
    }
    Ok(())
}

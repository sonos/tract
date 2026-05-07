//! Shared bench harness for WASM E2E model timing.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tract_nnef::internal::DimLike;
use tract_nnef::prelude::*;

pub type Runnable = Arc<TypedSimplePlan>;

pub fn build_zero_inputs(model: &Runnable) -> Result<TVec<TValue>> {
    let mut inputs = tvec![];
    let typed = model.model();
    for &outlet in typed.input_outlets()?.iter() {
        let fact = typed.outlet_fact(outlet)?;
        let shape: Vec<usize> =
            fact.shape.iter().map(|d| d.to_usize()).collect::<TractResult<Vec<_>>>()?;
        let dt = fact.datum_type;
        let tensor = Tensor::zero_dt(dt, &shape)?;
        inputs.push(tensor.into_tvalue());
    }
    Ok(inputs)
}

pub fn run_bench(
    model: &Runnable,
    inputs: &TVec<TValue>,
    warmup_iters: usize,
    timed_iters: usize,
    repetitions: usize,
) -> Result<Vec<f64>> {
    for _ in 0..warmup_iters {
        let _ = model.run(inputs.clone())?;
    }

    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let t0 = Instant::now();
        for _ in 0..timed_iters {
            let _ = model.run(inputs.clone())?;
        }
        let elapsed = t0.elapsed();
        let ns_per_call = elapsed.as_secs_f64() / timed_iters as f64 * 1e9;
        samples.push(ns_per_call);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples)
}

/// Print quality metrics for the model's output(s). Use after bench. With
/// fixed-shape models and a deterministic input (zeros), running this with
/// baseline vs relaxed-simd builds and comparing outputs is the quality
/// regression check (FMA gives ~1 ulp drift; mul+add is bit-stable).
pub fn run_quality_check(model: &Runnable, inputs: &TVec<TValue>) -> Result<()> {
    let outputs = model.run(inputs.clone())?;
    for (i, out) in outputs.iter().enumerate() {
        let dt = out.datum_type();
        let shape = out.shape();
        if dt == DatumType::F32 {
            let tensor: &Tensor = &*out;
            let slice: &[f32] = unsafe { tensor.as_slice_unchecked::<f32>() };
            let n = slice.len();
            let l2: f64 = slice.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            let mean: f64 = slice.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
            let preview: Vec<f32> = slice.iter().take(5).copied().collect();
            let last5: Vec<f32> =
                slice.iter().rev().take(5).copied().collect::<Vec<_>>().into_iter().rev().collect();
            // Cheap deterministic checksum: XOR of all bit-patterns
            let mut xor: u32 = 0;
            for x in slice {
                xor ^= x.to_bits();
            }
            eprintln!(
                "  output[{i}] shape={shape:?} dt=F32 n={n} L2={l2:.6e} mean={mean:.6e} xor=0x{xor:08x} first5={preview:?} last5={last5:?}"
            );
        } else {
            eprintln!("  output[{i}] shape={shape:?} dt={dt:?} (skip non-F32 stats)");
        }
    }
    Ok(())
}

pub fn print_stats(label: &str, samples: &[f64]) {
    let min = samples[0];
    let median = samples[samples.len() / 2];
    let max = samples[samples.len() - 1];
    let pct_spread = (max - min) / min * 100.0;
    let target = if cfg!(target_feature = "relaxed-simd") {
        "+relaxed-simd (FMA)"
    } else if cfg!(target_family = "wasm") {
        "+simd128 only (mul+add)"
    } else {
        "native"
    };
    eprintln!(
        "[{target}] {label}: min={min:.0} median={median:.0} max={max:.0} ns/inference (spread {pct_spread:.0}%, n={})",
        samples.len()
    );
}

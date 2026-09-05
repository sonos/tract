//! End-to-end latency for the RMSNorm GPU fusions in this PR: residual-add
//! absorption and gamma-scale absorption collapse a 3-dispatch-per-layer
//! pattern (Add -> RmsNorm -> Mul) into a single fused kernel dispatch.
//!
//! "unfused" reproduces the graph shape that existed on `main` before this
//! PR (no `ScaledRmsNorm`, no residual-fusion rule): three separate Metal
//! kernel dispatches per layer, chained the same way `fuse_rms_norm_scale`
//! and `fuse_rms_norm_residual` see them in a real transformer's residual
//! stream. "fused" is `RmsNorm::dispatch_eval`'s single `residual`+`scale`
//! call, which is exactly what those rewrite rules collapse the pattern
//! into.

use criterion::measurement::WallTime;
use criterion::*;
use tract_core::internal::*;
use tract_core::ops::math::{Add, Mul};
use tract_gpu::tensor::{DeviceTensor, IntoDevice};
use tract_metal::kernels::bin_ops;
use tract_metal::kernels::nn::rms_norm::RmsNorm;
use tract_metal::with_metal_stream;

const LAYERS: usize = 32;

struct Inputs {
    x0: DeviceTensor,
    deltas: Vec<DeviceTensor>,
    gammas: Vec<DeviceTensor>,
}

fn make_inputs(batch: usize, seq: usize, dim: usize) -> Inputs {
    let len = batch * seq * dim;
    let x0 = Tensor::from_shape(
        &[batch, seq, dim],
        &(0..len).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect::<Vec<_>>(),
    )
    .unwrap()
    .into_device()
    .unwrap();
    let deltas = (0..LAYERS)
        .map(|l| {
            Tensor::from_shape(
                &[batch, seq, dim],
                &(0..len).map(|i| ((i * 7 + l) % 19) as f32 / 64.0 - 0.15).collect::<Vec<_>>(),
            )
            .unwrap()
            .into_device()
            .unwrap()
        })
        .collect();
    let gammas = (0..LAYERS)
        .map(|l| {
            Tensor::from_shape(
                &[1, 1, dim],
                &(0..dim).map(|i| 0.5 + ((i + l) % 13) as f32 / 32.0).collect::<Vec<_>>(),
            )
            .unwrap()
            .into_device()
            .unwrap()
        })
        .collect();
    Inputs { x0, deltas, gammas }
}

fn unfused(crit: &mut BenchmarkGroup<WallTime>, label: &str, batch: usize, seq: usize, dim: usize) {
    with_metal_stream(|_| Ok(())).unwrap(); // one-time device/context init
    let inputs = make_inputs(batch, seq, dim);
    let eps = tensor0(1e-6f32);
    let shape = [batch, seq, dim];

    let run = || -> TractResult<()> {
        with_metal_stream(|stream| {
            let mut x = inputs.x0.clone();
            for l in 0..LAYERS {
                let h = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
                bin_ops::dispatch_eval(stream, &Add, &x, &inputs.deltas[l], &h)?;
                let n = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
                RmsNorm.dispatch_eval(stream, &h, None, None, 2, &eps, &n, None)?;
                let out = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
                bin_ops::dispatch_eval(stream, &Mul, &n, &inputs.gammas[l], &out)?;
                x = out;
            }
            stream.wait_until_completed()
        })
    };

    run().unwrap(); // warmup: pays one-time pipeline/library load cost
    crit.bench_function(format!("unfused_{label}"), |be| be.iter(|| run().unwrap()));
}

fn fused(crit: &mut BenchmarkGroup<WallTime>, label: &str, batch: usize, seq: usize, dim: usize) {
    with_metal_stream(|_| Ok(())).unwrap();
    let inputs = make_inputs(batch, seq, dim);
    let eps = tensor0(1e-6f32);
    let shape = [batch, seq, dim];

    let run = || -> TractResult<()> {
        with_metal_stream(|stream| {
            let mut x = inputs.x0.clone();
            for l in 0..LAYERS {
                let n = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
                let sum = DeviceTensor::uninitialized_dt(DatumType::F32, &shape)?;
                RmsNorm.dispatch_eval(
                    stream,
                    &x,
                    Some(&inputs.deltas[l]),
                    Some(&inputs.gammas[l]),
                    2,
                    &eps,
                    &n,
                    Some(&sum),
                )?;
                x = n;
            }
            stream.wait_until_completed()
        })
    };

    run().unwrap();
    crit.bench_function(format!("fused_{label}"), |be| be.iter(|| run().unwrap()));
}

/// Decode: 32 stacked norm layers, one token at a time (Qwen3.5-35B-ish
/// hidden size). Per-dispatch overhead dominates at this shape, so this is
/// where the fusion should show the largest relative win.
fn decode_step(c: &mut Criterion) {
    let mut g = c.benchmark_group("rms_norm_fusion_decode_b1_s1_d4096_l32");
    unfused(&mut g, "decode", 1, 1, 4096);
    fused(&mut g, "decode", 1, 1, 4096);
    g.finish();
}

/// Prefill: same 32 layers, a 32-token chunk. More real compute per
/// dispatch, so the fixed per-dispatch overhead is a smaller fraction of
/// total time here than in decode.
fn prefill_chunk(c: &mut Criterion) {
    let mut g = c.benchmark_group("rms_norm_fusion_prefill_b1_s32_d4096_l32");
    unfused(&mut g, "prefill", 1, 32, 4096);
    fused(&mut g, "prefill", 1, 32, 4096);
    g.finish();
}

criterion_group!(benches, decode_step, prefill_chunk);
criterion_main!(benches);

//! End-to-end Metal arena benchmark with a configurable active-expert path.
//!
//! Each configuration must run in its own process because the Metal context
//! and tuning profile are process-global.

use std::time::{Duration, Instant};

use tract_core::internal::*;
use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
use tract_core::ops::math::add;
use tract_core::ops::nn::{Softmax, SoftmaxKind};

fn build_model(layers: usize, experts: usize, m: usize, k: usize) -> TractResult<TypedModel> {
    let mut model = TypedModel::default();
    let mut x = model.add_source("x", f32::fact([1, m, k]))?;
    for layer in 0..layers {
        let mut combined = None;
        for expert in 0..experts {
            let seed = layer * experts + expert;
            let weights: Vec<f32> = (0..k * k)
                .map(|i| {
                    ((((i + seed * 7919) * 2654435761) % 2000) as f32 / 1000.0 - 1.0) / k as f32
                })
                .collect();
            let weights = model.add_const(
                format!("weights-{layer}-{expert}"),
                Tensor::from_shape(&[1, k, k], &weights)?,
            )?;
            let branch = model.wire_node(
                format!("matmul-{layer}-{expert}"),
                PrefixMatMul {
                    transpose_a: false,
                    transpose_b: false,
                    transpose_c: false,
                    quantize_output: None,
                    operating_dt: Some(DatumType::F32),
                },
                &[x, weights],
            )?[0];
            combined = Some(if let Some(previous) = combined {
                model.wire_node(format!("combine-{layer}-{expert}"), add(), &[previous, branch])?[0]
            } else {
                branch
            });
        }
        x = model.wire_node(
            format!("softmax-{layer}"),
            Softmax::new(tvec![2], None, SoftmaxKind::Softmax),
            &[combined.unwrap()],
        )?[0];
    }
    model.select_output_outlets(&[x])?;
    Ok(model)
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn env_usize(name: &str, default: usize) -> TractResult<usize> {
    std::env::var(name).ok().map_or(Ok(default), |value| value.parse().map_err(Into::into))
}

fn main() -> TractResult<()> {
    let layers = env_usize("ARENA_BENCH_LAYERS", 8)?;
    let experts = env_usize("ARENA_BENCH_EXPERTS", 3)?;
    let m = env_usize("ARENA_BENCH_M", 32)?;
    let k = env_usize("ARENA_BENCH_K", 2048)?;
    let warmup = env_usize("ARENA_BENCH_WARMUP", 3)?;
    let samples = env_usize("ARENA_BENCH_SAMPLES", 20)?;
    ensure!(experts > 0 && samples > 0, "experts and samples must be positive");

    let arena = std::env::var_os("TRACT_GPU_DISABLE_MEMORY_ARENA").is_none();
    let _metal = tract_metal::MetalTransform::default();
    let runtime = tract_core::runtime::runtime_for_name("metal")?
        .context("Metal runtime was not registered")?;
    let runnable = runtime.prepare(build_model(layers, experts, m, k)?)?;
    let mut state = runnable.spawn()?;
    let input = Tensor::from_shape(
        &[1, m, k],
        &(0..m * k).map(|i| ((i % 97) as f32 - 48.0) / 100.0).collect::<Vec<_>>(),
    )?
    .into_tvalue();

    let reference = state.run(tvec![input.clone()])?[0].clone().into_tensor().as_bytes().to_vec();
    for _ in 0..warmup {
        ensure!(state.run(tvec![input.clone()])?[0].clone().into_tensor().as_bytes() == reference);
    }
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let output = state.run(tvec![input.clone()])?;
        timings.push(start.elapsed());
        ensure!(output[0].clone().into_tensor().as_bytes() == reference);
    }
    let median = median(&mut timings);
    let mean = timings.iter().sum::<Duration>() / samples as u32;
    println!(
        "arena={arena} layers={layers} active_experts={experts} shape=1x{m}x{k} samples={samples} median_ms={:.3} mean_ms={:.3}",
        median.as_secs_f64() * 1e3,
        mean.as_secs_f64() * 1e3,
    );
    Ok(())
}

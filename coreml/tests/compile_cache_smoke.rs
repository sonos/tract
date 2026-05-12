//! Smoke test for the persistent compile cache.
//!
//! Builds a tiny synthetic Conv→Add model, runs `CoremlTransform` twice with
//! a clean cache the first time, and asserts:
//!
//!   1. The first call results in a cache **miss** + writes a `.mlmodelc` to
//!      `~/Library/Caches/tract-coreml/v1/<key>.mlmodelc/`.
//!   2. The second call results in a cache **hit**: the cached `.mlmodelc`
//!      directory is found and loaded directly (no `compileModelAtURL`).
//!   3. The second call is faster than the first (compile is skipped).
//!
//! Doesn't load any external model — purely synthetic, runs in-CI.

use anyhow::Result;
use half::f16;
use std::time::Instant;

use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::math;
use tract_core::ops::nn::DataFormat;
use tract_core::transform::ModelTransform;

use tract_coreml::CoremlTransform;
use tract_coreml::compile_cache::{CacheKey, cached_mlmodelc_path};
use tract_coreml::fusion;

const IN_C: usize = 4;
const OUT_C: usize = 8;
const H: usize = 8;
const W: usize = 8;
const K: usize = 3;

fn build_conv_add_model() -> Result<TypedModel> {
    // Use a UNIQUE weight seed (different from fusion_smoke's `0.07` /
    // `0.13` so this test never shares a cache key with concurrent tests.
    // Without this, cache_smoke's `remove_dir_all(cached)` races with
    // other tests' MLPackage loads that are mid-read on the same path.
    let weight: Vec<f16> = (0..(OUT_C * IN_C * K * K))
        .map(|i| f16::from_f32(((i as f32) * 0.31).sin() * 0.4))
        .collect();
    let bias: Vec<f16> =
        (0..OUT_C).map(|i| f16::from_f32(((i as f32) * 0.41).cos() * 0.15)).collect();
    let weight_tensor = Tensor::from_shape::<f16>(&[OUT_C, IN_C, K, K], &weight)?;
    let bias_tensor = Tensor::from_shape::<f16>(&[OUT_C], &bias)?;

    let pool_spec = PoolSpec {
        data_format: DataFormat::NCHW,
        kernel_shape: tvec![K, K],
        padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
        dilations: None,
        strides: None,
        input_channels: IN_C,
        output_channels: OUT_C,
    };
    let conv = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };

    let mut model = TypedModel::default();
    let input = model.add_source("input", f16::fact([1usize, IN_C, H, W]))?;
    let weight_outlet = model.add_const("conv_weight", weight_tensor)?;
    let bias_outlet = model.add_const("conv_bias", bias_tensor)?;
    let conv_out = model.wire_node("conv", conv, &[input, weight_outlet, bias_outlet])?[0];
    let _add_out = model.wire_node("add_self", math::add(), &[conv_out, conv_out])?[0];
    model.auto_outputs()?;
    Ok(model)
}

#[test]
fn cache_misses_then_hits_on_second_transform() -> Result<()> {
    let model = build_conv_add_model()?;

    // Compute the cache key for the (single) subgraph this model produces, and
    // wipe the cache entry if present so we observe a deterministic miss.
    let subgraphs = fusion::identify_subgraphs(&model)?;
    assert_eq!(subgraphs.len(), 1, "expected a single subgraph for Conv+Add");
    let (_io, model_bytes, weight_bytes) = fusion::build_subgraph_artifacts(&model, &subgraphs[0])?;
    let key = CacheKey::compute(&model_bytes, &weight_bytes);
    let cached = cached_mlmodelc_path(&key).expect("cache path");
    if cached.exists() {
        std::fs::remove_dir_all(&cached)?;
    }
    assert!(!cached.exists(), "precondition: cache entry absent");

    // First transform should compile + populate the cache.
    let t0 = Instant::now();
    {
        let mut m = model.clone();
        CoremlTransform::default().transform(&mut m)?;
    }
    let cold = t0.elapsed();
    assert!(cached.exists(), "first transform should have populated the cache");

    // Second transform should hit the cache (skip compile).
    let t1 = Instant::now();
    {
        let mut m = model.clone();
        CoremlTransform::default().transform(&mut m)?;
    }
    let warm = t1.elapsed();

    println!(
        "[cache_smoke] cold={cold:?} warm={warm:?}  ratio={:.2}x",
        cold.as_secs_f64() / warm.as_secs_f64()
    );

    // Warm should be at least 2× faster than cold. (We typically see 5-10× on
    // real models; even a tiny synthetic one should manage 2× because Apple's
    // compileModelAtURL takes ~50–100 ms minimum and a cache hit skips it
    // entirely.)
    assert!(
        warm < cold / 2,
        "expected warm ({warm:?}) < cold/2 ({cold:?}/2); cache may not be hitting"
    );
    Ok(())
}

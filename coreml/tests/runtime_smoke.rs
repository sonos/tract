//! End-to-end smoke test for `CoremlRuntime` + `CoremlTransform`.
//!
//! Builds a synthetic tract typed model:
//!     input (F16 [1,4,8,8]) → Conv (3×3, pad=1, F16 weight, F16 bias) → output (F16 [1,8,8,8])
//!
//! Runs it two ways:
//!   1. **CPU**: `model.into_runnable().run(...)` — tract's standard CPU path.
//!   2. **CoreML**: `CoremlRuntime.prepare(model).run(...)` — our new path,
//!      which applies `CoremlTransform` (replacing the Conv with a `CoremlOp`)
//!      and then runs the result via `TypedSimplePlan`.
//!
//! Asserts the two outputs match within FP16 tolerance. This proves the whole
//! Phase 1 stack works through the user-facing `Runtime` API.

use std::sync::Arc;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::transform::ModelTransform;

use tract_coreml::{CoremlRuntime, CoremlTransform};

const IN_C: usize = 4;
const OUT_C: usize = 8;
const H: usize = 8;
const W: usize = 8;
const K: usize = 3; // kernel size

/// Build the test model: input → Conv → output.
fn build_model() -> Result<TypedModel> {
    // Deterministic small-magnitude weights + bias so FP16 doesn't saturate.
    let weight_data: Vec<f16> = (0..(OUT_C * IN_C * K * K))
        .map(|i| f16::from_f32(((i as f32) * 0.07).sin() * 0.3))
        .collect();
    let bias_data: Vec<f16> =
        (0..OUT_C).map(|i| f16::from_f32(((i as f32) * 0.13).cos() * 0.1)).collect();
    let weight_tensor = Tensor::from_shape::<f16>(&[OUT_C, IN_C, K, K], &weight_data)?;
    let bias_tensor = Tensor::from_shape::<f16>(&[OUT_C], &bias_data)?;

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
    let weight = model.add_const("conv_weight", weight_tensor)?;
    let bias = model.add_const("conv_bias", bias_tensor)?;
    let _conv_out = model.wire_node("conv", conv, &[input, weight, bias])?[0];
    model.auto_outputs()?;
    Ok(model)
}

/// Deterministic input tensor.
fn make_input() -> Result<Tensor> {
    let data: Vec<f16> =
        (0..(IN_C * H * W)).map(|i| f16::from_f32(((i as f32) * 0.04).cos() * 0.5)).collect();
    Tensor::from_shape::<f16>(&[1, IN_C, H, W], &data)
}

#[test]
fn cpu_vs_coreml_runtime_match_within_fp16_tol() -> Result<()> {
    let input_tensor = make_input()?;
    let model_cpu = build_model()?;
    let model_coreml = build_model()?;

    // 1. CPU baseline.
    let cpu_runnable = model_cpu.into_runnable()?;
    let cpu_outputs = cpu_runnable.run(tvec![input_tensor.clone().into()])?;
    assert_eq!(cpu_outputs.len(), 1);
    let cpu_out = &cpu_outputs[0];
    assert_eq!(cpu_out.shape(), &[1, OUT_C, H, W]);
    assert_eq!(cpu_out.datum_type(), DatumType::F16);

    // 2. CoreML path via the Runtime API.
    let runtime = CoremlRuntime;
    let coreml_runnable = runtime.prepare(model_coreml)?;
    let coreml_outputs = coreml_runnable.run(tvec![input_tensor.into()])?;
    assert_eq!(coreml_outputs.len(), 1);
    let coreml_out = &coreml_outputs[0];
    assert_eq!(coreml_out.shape(), &[1, OUT_C, H, W]);
    assert_eq!(coreml_out.datum_type(), DatumType::F16);

    // Compare element-wise within FP16 tolerance.
    let cpu_slice = unsafe { cpu_out.as_slice_unchecked::<f16>() };
    let coreml_slice = unsafe { coreml_out.as_slice_unchecked::<f16>() };
    assert_eq!(cpu_slice.len(), coreml_slice.len());

    let mut max_abs_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (c, k) in cpu_slice.iter().zip(coreml_slice.iter()) {
        let cf = c.to_f32();
        let kf = k.to_f32();
        let d = (cf - kf).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (cf as f64) * (cf as f64);
    }
    let rel_l2 = (sum_sq_diff / sum_sq_ref).sqrt();

    println!(
        "[cpu vs coreml] max_abs_diff = {max_abs_diff:.6}  rel_l2 = {rel_l2:.6e}  (FP16 tol: max_abs<5e-3, rel_l2<1e-2)"
    );

    assert!(max_abs_diff < 5e-3, "max_abs_diff {max_abs_diff} exceeds FP16 tolerance");
    assert!(rel_l2 < 1e-2, "rel_l2 {rel_l2} exceeds FP16 tolerance");
    Ok(())
}

/// Sanity-check that `CoremlTransform` actually replaced the Conv with a
/// CoremlOp (vs silently no-op'ing). We don't want a green test that's
/// just running CPU on both sides.
#[test]
fn transform_actually_replaces_conv() -> Result<()> {
    let mut model = build_model()?;

    // Pre-transform sanity: there's a Conv node.
    let conv_node_count_before = model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    assert_eq!(conv_node_count_before, 1, "expected exactly one Conv node before transform");

    CoremlTransform::default().transform(&mut model)?;

    // Post-transform: the Conv must be gone, replaced by a CoremlOp.
    let conv_node_count_after = model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    assert_eq!(conv_node_count_after, 0, "Conv node should have been replaced by CoremlOp");

    let coreml_node_count =
        model.nodes.iter().filter(|n| n.op_as::<tract_coreml::CoremlOp>().is_some()).count();
    assert_eq!(coreml_node_count, 1, "expected exactly one CoremlOp after transform");
    Ok(())
}

// Silence the unused `Arc` warning if this test is the only consumer.
#[allow(dead_code)]
fn _force_arc_use() -> Option<Arc<()>> {
    None
}

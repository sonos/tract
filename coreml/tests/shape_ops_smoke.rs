//! Synthetic smoke tests for shape-manipulation translators: Reshape,
//! MoveAxis (Transpose), Pad.
//!
//! Each test wires a tiny tract IR fragment that exercises one translator,
//! runs CPU vs CoreML in-process, and asserts FP16 outputs match within
//! tolerance. No external models needed — surfaces translator regressions
//! in CI before they hit a real model.
//!
//! These translators landed across Phase 4 (defrag pass for SAM 2 Hiera-Tiny)
//! but didn't get unit coverage at the time. This file plugs that gap.

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::transform::ModelTransform;

use tract_coreml::{CoremlOp, CoremlTransform};

fn fill_sin(n: usize, seed: f32) -> Vec<f16> {
    (0..n).map(|i| f16::from_f32(((i as f32 + seed) * 0.07).sin() * 0.5)).collect()
}

fn compare(cpu: &[f16], coreml: &[f16], label: &str) {
    assert_eq!(cpu.len(), coreml.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (cf, kf) in cpu.iter().zip(coreml.iter()) {
        let cv = cf.to_f32();
        let kv = kf.to_f32();
        let d = (cv - kv).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (cv as f64) * (cv as f64);
    }
    let rel_l2 = (sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt();
    println!("[{label}] max_abs = {max_abs:.6}  rel_l2 = {rel_l2:.6e}");
    assert!(max_abs < 5e-3, "{label}: max_abs {max_abs} too high");
    assert!(rel_l2 < 1e-2, "{label}: rel_l2 {rel_l2} too high");
}

/// Wraps `body` (which appends ops to the model after a Conv producing a
/// rank-4 tensor) into a model that exercises Conv → body → Conv. Returns
/// the model and the input tensor to use. The two Convs sandwich the body
/// so the body sits inside a CoremlOp subgraph (Conv is a known-good
/// translator that defines the in-MLPackage rank-4 convention).
fn build_sandwich_model(
    in_shape: [usize; 4],
    body: impl FnOnce(&mut TypedModel, OutletId) -> Result<OutletId>,
) -> Result<(TypedModel, Tensor)> {
    let [n, c_in, h, w] = in_shape;
    let weight: Vec<f16> =
        (0..c_in * c_in).map(|i| f16::from_f32(((i as f32) * 0.13).sin() * 0.3)).collect();
    let bias: Vec<f16> =
        (0..c_in).map(|i| f16::from_f32(((i as f32) * 0.07).cos() * 0.1)).collect();
    let weight_tensor = Tensor::from_shape::<f16>(&[c_in, c_in, 1, 1], &weight)?;
    let bias_tensor = Tensor::from_shape::<f16>(&[c_in], &bias)?;
    let pool_spec = PoolSpec {
        data_format: DataFormat::NCHW,
        kernel_shape: tvec![1, 1],
        padding: PaddingSpec::Explicit(tvec![0, 0], tvec![0, 0]),
        dilations: None,
        strides: None,
        input_channels: c_in,
        output_channels: c_in,
    };
    let conv = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group: 1, q_params: None };

    let mut model = TypedModel::default();
    let input = model.add_source("input", f16::fact([n, c_in, h, w]))?;
    let weight_outlet = model.add_const("conv_weight", weight_tensor)?;
    let bias_outlet = model.add_const("conv_bias", bias_tensor)?;
    let conv_out = model.wire_node("conv1", conv.clone(), &[input, weight_outlet, bias_outlet])?[0];
    let _body_out = body(&mut model, conv_out)?;
    model.auto_outputs()?;

    let n_in: usize = in_shape.iter().product();
    let in_data = fill_sin(n_in, 0.0);
    let in_tensor = Tensor::from_shape::<f16>(&in_shape, &in_data)?;
    Ok((model, in_tensor))
}

fn run_cpu_vs_coreml(model: TypedModel, input: Tensor, label: &str) -> Result<usize> {
    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("[{label}] CoremlOps: {coreml_count}");

    let cpu_out = cpu_model.into_runnable()?.run(tvec![input.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![input.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, label);
    Ok(coreml_count)
}

// =========================================================================
// Reshape
// =========================================================================

/// Conv → Reshape (rank 4 → rank 3, collapse spatial). Verifies the
/// translator handles the common case of flattening spatial dims for a
/// downstream Linear layer.
#[test]
fn reshape_rank4_to_rank3_collapse_spatial() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        // Reshape from [1, 8, 4, 4] → [1, 8, 16] (collapse last 2 dims).
        let reshape = AxisOp::Reshape(2, tvec![4.to_dim(), 4.to_dim()], tvec![16.to_dim()]);
        Ok(m.wire_node("reshape", reshape, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "reshape_rank4_to_rank3")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Reshape fused)");
    Ok(())
}

/// Reshape that splits an axis: [1, 8, 16] → [1, 8, 4, 4]. Common in
/// attention block windowing.
#[test]
fn reshape_split_axis() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        // First collapse to rank 3, then split back.
        let reshape1 = AxisOp::Reshape(2, tvec![4.to_dim(), 4.to_dim()], tvec![16.to_dim()]);
        let r1 = m.wire_node("reshape1", reshape1, &[conv_out])?[0];
        let reshape2 = AxisOp::Reshape(2, tvec![16.to_dim()], tvec![4.to_dim(), 4.to_dim()]);
        Ok(m.wire_node("reshape2", reshape2, &[r1])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "reshape_split_axis")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + 2× Reshape fused)");
    Ok(())
}

// =========================================================================
// MoveAxis (Transpose)
// =========================================================================

/// Conv → MoveAxis swap last two dims. Common in transformer attention
/// for bringing head dim to the right position.
#[test]
fn move_axis_swap_last_two() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        // Move axis 2 → 3 (swap H and W).
        Ok(m.wire_node("move", AxisOp::Move(2, 3), &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "move_axis_swap_last_two")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + MoveAxis fused)");
    Ok(())
}

/// Two MoveAxis in sequence (model attention's `BHSD → BSHD → BHSD` pattern).
#[test]
fn move_axis_round_trip() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        let m1 = m.wire_node("move1", AxisOp::Move(1, 2), &[conv_out])?[0];
        Ok(m.wire_node("move2", AxisOp::Move(2, 1), &[m1])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "move_axis_round_trip")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + 2× MoveAxis fused)");
    Ok(())
}

// =========================================================================
// Pad
// =========================================================================

/// Conv → Pad (constant 0). Common before windowed-attention to make
/// spatial dim divisible by window size.
#[test]
fn pad_constant_zero() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        let pad = Pad {
            pads: vec![(0, 0), (0, 0), (1, 1), (1, 1)],
            mode: PadMode::Constant(std::sync::Arc::new(f16::from_f32(0.0).into())),
        };
        Ok(m.wire_node("pad", pad, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "pad_constant_zero")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Pad fused)");
    Ok(())
}

/// Pad with reflect mode. Used in some image-preprocess pipelines.
#[test]
fn pad_reflect_mode() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        let pad = Pad { pads: vec![(0, 0), (0, 0), (1, 1), (1, 1)], mode: PadMode::Reflect };
        Ok(m.wire_node("pad", pad, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "pad_reflect_mode")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Pad fused)");
    Ok(())
}

// =========================================================================
// RmsNorm (standalone, not via LayerNorm fold)
// =========================================================================

/// Standalone RmsNorm — not absorbed by LayerNorm fold (no Sub/mean before).
/// Used in modern LLMs (LLaMA, Mistral) and surfaces inside tract's LayerNorm
/// chain. Translator decomposes to square + reduce_mean + add(eps) + rsqrt + mul.
#[test]
fn rms_norm_standalone() -> Result<()> {
    use tract_core::ops::nn::RmsNorm;

    const B: usize = 2;
    const S: usize = 4;
    const D: usize = 8;

    let mut model = TypedModel::default();
    let input = model.add_source("input", f16::fact([B, S, D]))?;
    let eps_t = Tensor::from(1e-5f32);
    let rms = RmsNorm { axis: 2, eps: std::sync::Arc::new(eps_t) };
    model.wire_node("rms", rms, &[input])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected single CoremlOp (standalone RmsNorm)");

    let raw: Vec<f16> = fill_sin(B * S * D, 5.0);
    let input_t = Tensor::from_shape::<f16>(&[B, S, D], &raw)?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![input_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![input_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    // RmsNorm involves reduce + rsqrt + mul; FP16 noise is slightly higher.
    let mut max_abs = 0.0f32;
    for (cf, kf) in c.iter().zip(k.iter()) {
        let d = (cf.to_f32() - kf.to_f32()).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    println!("[rms_norm_standalone] max_abs = {max_abs:.6}");
    assert!(max_abs < 5e-3, "rms_norm_standalone: max_abs {max_abs} too high");
    Ok(())
}

// =========================================================================
// Slice
// =========================================================================

/// Conv → Slice on H axis. Common in ROI extraction and windowed-attention
/// preparation. Verifies the slice translator handles non-trivial begin/end.
#[test]
fn slice_h_axis() -> Result<()> {
    use tract_core::ops::array::Slice;
    let (model, input) = build_sandwich_model([1, 8, 8, 8], |m, conv_out| {
        // Slice H axis from [2, 6) — reduces H from 8 to 4.
        let slice = Slice { axis: 2, start: 2.to_dim(), end: 6.to_dim() };
        Ok(m.wire_node("slice", slice, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "slice_h_axis")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Slice fused)");
    Ok(())
}

// =========================================================================
// Concat
// =========================================================================

/// Conv → Concat(self, self) on channel axis. Tests the concat translator's
/// handling of variadic inputs + per-input MLPackage shape derivation.
#[test]
fn concat_channel_self_self() -> Result<()> {
    use tract_core::ops::array::TypedConcat;
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        // Concat conv_out with itself along channel axis (axis 1) → [1, 16, 4, 4].
        let concat = TypedConcat { axis: 1 };
        Ok(m.wire_node("concat", concat, &[conv_out, conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "concat_channel_self_self")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Concat fused)");
    Ok(())
}

// =========================================================================
// AddAxis / RmAxis (singleton-dim insert/remove)
// =========================================================================

/// Conv → AddAxis(0) → RmAxis(0). Tests the AddAxis + RmAxis pair commonly
/// appearing around batch-dim manipulation.
#[test]
fn add_axis_rm_axis_round_trip() -> Result<()> {
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        let added = m.wire_node("add", AxisOp::Add(0), &[conv_out])?[0];
        Ok(m.wire_node("rm", AxisOp::Rm(0), &[added])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "add_axis_rm_axis_round_trip")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + AddAxis + RmAxis fused)");
    Ok(())
}

// =========================================================================
// Cast (F16 → F16 no-op as well as actual dtype changes)
// =========================================================================

/// Synthetic Cast(F16 → F16) — the most common case after the f32_to_f16
/// transform leaves redundant Casts. Verifies the translator handles
/// no-op casts cleanly.
#[test]
fn cast_f16_to_f16_noop() -> Result<()> {
    use tract_core::ops::cast;
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        Ok(m.wire_node("cast", cast::cast(DatumType::F16), &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "cast_f16_to_f16_noop")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Cast fused)");
    Ok(())
}

// =========================================================================
// EinSum-as-1×1-conv (NIHW,OI->NOHW pattern from MobileNet-class models)
// =========================================================================

// =========================================================================
// Reduce variants (Sum, Max, Min, Prod, MeanOfSquares)
// =========================================================================

fn reduce_test(reducer: tract_core::ops::nn::Reducer, label: &str) -> Result<()> {
    use tract_core::ops::nn::Reduce;
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        Ok(m.wire_node("reduce", Reduce { axes: tvec![2, 3], reducer }, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, label)?;
    assert_eq!(count, 1, "{label}: expected single CoremlOp (Conv + Reduce fused)");
    Ok(())
}

#[test]
fn reduce_sum() -> Result<()> {
    reduce_test(tract_core::ops::nn::Reducer::Sum, "reduce_sum")
}
#[test]
fn reduce_max() -> Result<()> {
    reduce_test(tract_core::ops::nn::Reducer::Max, "reduce_max")
}
#[test]
fn reduce_min() -> Result<()> {
    reduce_test(tract_core::ops::nn::Reducer::Min, "reduce_min")
}
#[test]
fn reduce_mean_of_squares() -> Result<()> {
    reduce_test(tract_core::ops::nn::Reducer::MeanOfSquares, "reduce_mean_of_squares")
}

// =========================================================================
// Activation variants (Sigmoid, Tanh, Silu, GeluApproximate, Erf, Rsqrt, Sqrt, Square)
// =========================================================================

fn activation_test(activation: &'static str) -> Result<()> {
    use tract_core::ops::element_wise::ElementWiseOp;
    let mini_op: Box<dyn tract_core::ops::element_wise::ElementWiseMiniOp> = match activation {
        "Sigmoid" => Box::new(tract_core::ops::nn::Sigmoid {}),
        "Tanh" => Box::new(tract_core::ops::math::Tanh {}),
        "Silu" => Box::new(tract_core::ops::nn::Silu {}),
        "GeluApproximate" => Box::new(tract_core::ops::nn::GeluApproximate { fast_impl: false }),
        "Erf" => Box::new(tract_core::ops::math::Erf {}),
        "Square" => Box::new(tract_core::ops::math::Square {}),
        "Ln" => Box::new(tract_core::ops::math::Ln {}),
        "Recip" => Box::new(tract_core::ops::math::Recip {}),
        "LeakyRelu" => Box::new(tract_core::ops::nn::LeakyRelu { alpha: 0.01 }),
        "Neg" => Box::new(tract_core::ops::math::Neg {}),
        "Cos" => Box::new(tract_core::ops::math::Cos {}),
        "Sin" => Box::new(tract_core::ops::math::Sin {}),
        other => panic!("test fixture missing for activation '{other}'"),
    };
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        Ok(m.wire_node(activation, ElementWiseOp(mini_op, None), &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, &format!("activation_{activation}"))?;
    assert_eq!(count, 1);
    Ok(())
}

#[test]
fn activation_sigmoid() -> Result<()> {
    activation_test("Sigmoid")
}
#[test]
fn activation_tanh() -> Result<()> {
    activation_test("Tanh")
}
#[test]
fn activation_silu() -> Result<()> {
    activation_test("Silu")
}
#[test]
fn activation_gelu_approximate() -> Result<()> {
    activation_test("GeluApproximate")
}
#[test]
fn activation_erf() -> Result<()> {
    activation_test("Erf")
}
#[test]
fn activation_square() -> Result<()> {
    activation_test("Square")
}
#[test]
fn activation_leaky_relu() -> Result<()> {
    activation_test("LeakyRelu")
}
#[test]
fn activation_neg() -> Result<()> {
    activation_test("Neg")
}
#[test]
fn activation_cos() -> Result<()> {
    activation_test("Cos")
}
#[test]
fn activation_sin() -> Result<()> {
    activation_test("Sin")
}
// Ln / Recip blow up near 0, so the plain conv→activation fixture is
// numerically degenerate (conv output passes through 0). Add a +2.0 bias
// const before the activation to keep inputs safely positive and bounded
// away from zero — still exercises the same translator + emit path.
fn activation_test_strict_positive(activation: &'static str) -> Result<()> {
    use tract_core::ops::element_wise::ElementWiseOp;
    use tract_core::ops::math;
    let mini_op: Box<dyn tract_core::ops::element_wise::ElementWiseMiniOp> = match activation {
        "Ln" => Box::new(math::Ln {}),
        "Recip" => Box::new(math::Recip {}),
        other => panic!("test fixture missing for strict-positive activation '{other}'"),
    };
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        let bias_t = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::from_f32(2.0)])?;
        let bias_outlet = m.add_const(format!("{activation}_bias"), bias_t)?;
        let biased =
            m.wire_node(format!("{activation}_add_bias"), math::add(), &[conv_out, bias_outlet])?
                [0];
        Ok(m.wire_node(activation, ElementWiseOp(mini_op, None), &[biased])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, &format!("activation_{activation}"))?;
    assert_eq!(count, 1);
    Ok(())
}

#[test]
fn activation_ln() -> Result<()> {
    activation_test_strict_positive("Ln")
}
#[test]
fn activation_recip() -> Result<()> {
    activation_test_strict_positive("Recip")
}

// =========================================================================
// Div (BinOp variant) — added for Parakeet preprocessor
// =========================================================================

/// Conv → Div by const. The real_div MIL op should handle this.
#[test]
fn binop_div_by_const() -> Result<()> {
    use tract_core::ops::math;
    let (model, input) = build_sandwich_model([1, 8, 4, 4], |m, conv_out| {
        // Const denominator (avoid /0 — pick something safe).
        let denom_data: Vec<f16> = (0..8).map(|i| f16::from_f32(1.0 + 0.1 * (i as f32))).collect();
        let denom_t = Tensor::from_shape::<f16>(&[1, 8, 1, 1], &denom_data)?;
        let denom = m.add_const("denom", denom_t)?;
        Ok(m.wire_node("div", math::div(), &[conv_out, denom])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "binop_div_by_const")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Div fused)");
    Ok(())
}

// =========================================================================
// SumPool / MaxPool with SameUpper padding (added for InceptionV3)
// =========================================================================

/// Conv → SumPool(normalize=true) with `SameUpper` padding (3×3 kernel,
/// stride 1) — the exact pattern that surfaces 9× in InceptionV3's
/// Inception modules. Pre-fix every one of these stranded with
/// "SumPool padding SameUpper"; post-fix they map to MIL `avg_pool`
/// `pad_type="same"`.
#[test]
fn sumpool_normalized_same_upper() -> Result<()> {
    use tract_core::ops::cnn::SumPool;
    let (model, input) = build_sandwich_model([1, 8, 6, 6], |m, conv_out| {
        let pool_spec = PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec![3, 3],
            padding: PaddingSpec::SameUpper,
            dilations: None,
            strides: None,
            input_channels: 8,
            output_channels: 8,
        };
        let sp = SumPool { pool_spec, count_include_pad: false, normalize: true };
        Ok(m.wire_node("avgpool_same", sp, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "sumpool_normalized_same_upper")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + AvgPool fused)");
    Ok(())
}

/// Conv → SumPool(normalize=true) with `ExplicitOnnxPool(_, _, ceil_mode=true)`
/// padding. Exercises a 2×2 / stride-2 pool over a 5×5 spatial input where
/// floor- and ceil-mode produce different output dims (floor = 2, ceil = 3).
/// RVM hits this with a 2×2/stride-2 pool over 45×60 — without ceil_mode
/// propagation the MIL output dim is off-by-one and downstream concat fails
/// MPS validation. Pre-fix the test would have stranded with
/// "SumPool padding ExplicitOnnxPool"; post-fix it lands as a single
/// CoremlOp and matches CPU output bit-for-bit.
#[test]
fn sumpool_normalized_explicit_onnx_pool_ceil_mode() -> Result<()> {
    use tract_core::ops::cnn::SumPool;
    let (model, input) = build_sandwich_model([1, 8, 5, 5], |m, conv_out| {
        let pool_spec = PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec![2, 2],
            padding: PaddingSpec::ExplicitOnnxPool(tvec![0, 0], tvec![0, 0], true),
            dilations: None,
            strides: Some(tvec![2, 2]),
            input_channels: 8,
            output_channels: 8,
        };
        let sp = SumPool { pool_spec, count_include_pad: false, normalize: true };
        Ok(m.wire_node("avgpool_ceil", sp, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "sumpool_explicit_onnx_pool_ceil_mode")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + AvgPool fused)");
    Ok(())
}

/// Conv → MaxPool with `SameUpper` padding (3×3 kernel, stride 1).
#[test]
fn maxpool_same_upper() -> Result<()> {
    use tract_core::ops::cnn::MaxPool;
    let (model, input) = build_sandwich_model([1, 8, 6, 6], |m, conv_out| {
        let pool_spec = PoolSpec {
            data_format: DataFormat::NCHW,
            kernel_shape: tvec![3, 3],
            padding: PaddingSpec::SameUpper,
            dilations: None,
            strides: None,
            input_channels: 8,
            output_channels: 8,
        };
        let mp = MaxPool { pool_spec, with_index_outputs: None };
        Ok(m.wire_node("maxpool_same", mp, &[conv_out])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "maxpool_same_upper")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + MaxPool fused)");
    Ok(())
}

/// Synthetic 1×1 conv via EinSum. tract emits this signature for some
/// declutter rewrites; we route it through the same conv emitter as
/// AxisOp::Reshape's translator.
#[test]
fn einsum_as_1x1_conv_nihw_oi_to_nohw() -> Result<()> {
    use tract_core::ops::einsum::EinSum;
    const N: usize = 1;
    const I: usize = 8;
    const H: usize = 4;
    const W: usize = 4;
    const O: usize = 16;

    let mut model = TypedModel::default();
    let input = model.add_source("input", f16::fact([N, I, H, W]))?;
    // Weight: [O, I] for the einsum NIHW,OI->NOHW.
    let weight_data: Vec<f16> =
        (0..O * I).map(|i| f16::from_f32(((i as f32) * 0.13).sin() * 0.3)).collect();
    let weight = Tensor::from_shape::<f16>(&[O, I], &weight_data)?;
    let weight_outlet = model.add_const("w", weight)?;
    let es = EinSum::new("NIHW,OI->NOHW".parse()?, DatumType::F16);
    model.wire_node("einsum", es, &[input, weight_outlet])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected single CoremlOp (EinSum-as-1×1-conv)");

    let in_n = N * I * H * W;
    let in_data = fill_sin(in_n, 0.0);
    let in_t = Tensor::from_shape::<f16>(&[N, I, H, W], &in_data)?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![in_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![in_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, "einsum_as_1x1_conv");
    Ok(())
}

// =========================================================================
// Gather (added Phase 5 for LLM embedding lookup + ViT CLS-token pluck)
// =========================================================================

/// LLM-style embedding lookup: const F16 embedding matrix `[V, H]` indexed
/// by an i64 input `[B, S]` of token IDs → output `[B, S, H]`. Mirrors the
/// stranded `Gather` at position 0 of every SmolLM2-class decoder LLM.
#[test]
fn gather_embedding_lookup() -> Result<()> {
    use tract_core::ops::array::Gather;
    const V: usize = 32;
    const H: usize = 8;
    const B: usize = 1;
    const S: usize = 4;

    let mut model = TypedModel::default();
    // Embedding matrix as a Const.
    let emb_data: Vec<f16> =
        (0..V * H).map(|i| f16::from_f32(((i as f32) * 0.07).sin() * 0.5)).collect();
    let emb_t = Tensor::from_shape::<f16>(&[V, H], &emb_data)?;
    let emb = model.add_const("embedding_matrix", emb_t)?;
    // Token-id input as a runtime I64 source. tract's Gather op requires
    // I64 indices on the CPU side, and ONNX LLM exports declare token-id
    // inputs as I64. CoremlOp's eval narrows I64 → I32 at the MIL boundary
    // (MLMultiArray doesn't natively support Int64).
    let ids = model.add_source("token_ids", TypedFact::dt_shape(DatumType::I64, [B, S]))?;
    model.wire_node("gather", Gather::new(0), &[emb, ids])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected single CoremlOp for the lone Gather");

    // Pick a couple of token IDs and verify CPU vs CoreML agree.
    let token_ids: Vec<i64> = vec![0, 5, 17, 31];
    let ids_t = Tensor::from_shape::<i64>(&[B, S], &token_ids)?;
    let cpu_out = cpu_model.into_runnable()?.run(tvec![ids_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![ids_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, "gather_embedding_lookup");
    Ok(())
}

/// LLM-style mask construction: `Eq(positions, padding_id)` produces a bool
/// mask, then `Iff(mask, neg_inf, scores)` substitutes the masked entries.
/// Mirrors the SmolLM2 attention-mask path where Eq + Iff fragmented the
/// model into 2 CoremlOps until both got translators.
#[test]
fn eq_iff_attention_mask_chain() -> Result<()> {
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::logic::{Iff, comp_eq};
    // Realistic attention-mask shape: positions and scores share the
    // (B, H, S, S) layout. Mask broadcasts H from 1 → H_full.
    const B: usize = 1;
    const S: usize = 4;
    const H: usize = 2;

    let mut model = TypedModel::default();
    let positions = model.add_source("positions", f16::fact([B, 1, S, S]))?;
    let scores = model.add_source("scores", f16::fact([B, H, S, S]))?;
    // Compare to a const padding-id (here, 0.0).
    let pad_t = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::from_f32(0.0)])?;
    let pad = model.add_const("padding_id", pad_t)?;
    let mask = model.wire_node("mask", TypedBinOp(comp_eq(), None), &[positions, pad])?[0];
    // Const "neg infinity" replacement value.
    let neg_inf_t = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::from_f32(-65504.0)])?;
    let neg_inf = model.add_const("neg_inf", neg_inf_t)?;
    model.wire_node("mask_apply", Iff::default(), &[mask, neg_inf, scores])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected single CoremlOp (Eq + Iff fused)");

    // Run with positions where some entries are 0 (will get masked to -inf).
    let pos_data: Vec<f16> = (0..B * S * S)
        .map(|i| f16::from_f32(if i % 3 == 0 { 0.0 } else { 0.5 + (i as f32) * 0.1 }))
        .collect();
    let scores_data: Vec<f16> =
        (0..B * H * S * S).map(|i| f16::from_f32(((i as f32) * 0.1).sin() * 0.5)).collect();
    let pos_t = Tensor::from_shape::<f16>(&[B, 1, S, S], &pos_data)?;
    let scores_t = Tensor::from_shape::<f16>(&[B, H, S, S], &scores_data)?;
    let cpu_out =
        cpu_model.into_runnable()?.run(tvec![pos_t.clone().into(), scores_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![pos_t.into(), scores_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, "eq_iff_attention_mask_chain");
    Ok(())
}

/// ViT-style CLS-token pluck: runtime F16 encoder output `[1, P+1, H]`
/// indexed by a scalar i64 const `0` → output `[1, H]`.
#[test]
fn gather_cls_pluck() -> Result<()> {
    use tract_core::ops::array::Gather;
    let (model, input) = build_sandwich_model([1, 8, 1, 4], |m, conv_out| {
        // Scalar Const indices = 0 (rank-0 i64).
        let idx_t = Tensor::from_shape::<i64>(&[], &[0])?;
        let idx = m.add_const("cls_index", idx_t)?;
        // Gather along axis 1 (the patch dim, size = 8 here).
        Ok(m.wire_node("cls_pluck", Gather::new(1), &[conv_out, idx])?[0])
    })?;
    let count = run_cpu_vs_coreml(model, input, "gather_cls_pluck")?;
    assert_eq!(count, 1, "expected single CoremlOp (Conv + Gather fused)");
    Ok(())
}

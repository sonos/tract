//! Synthetic smoke tests for transformer-prep translators: Softmax,
//! general MatMul (runtime × runtime + Linear), LayerNorm.
//!
//! These don't require any external model — each test wires a tiny tract
//! IR fragment that exercises one translator, runs CPU vs CoreML in-process,
//! and asserts the FP16 outputs match within tolerance.
//!
//! Goal: surface translator bugs at CI time without needing SAM 3 weights.
//! When SAM 3 lands, these tests give us a baseline to attribute regressions
//! to.

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math;
use tract_core::ops::nn::{Softmax, SoftmaxKind};
use tract_core::transform::ModelTransform;

use tract_coreml::{CoremlOp, CoremlTransform};

const TOL_MAX_ABS: f32 = 5e-3;
const TOL_REL_L2: f64 = 1e-2;

fn fill_sin(n: usize, seed: f32) -> Vec<f16> {
    (0..n).map(|i| f16::from_f32(((i as f32 + seed) * 0.07).sin() * 0.5)).collect()
}

fn compare(cpu: &[f16], coreml: &[f16], label: &str) {
    compare_tol(cpu, coreml, label, TOL_MAX_ABS, TOL_REL_L2)
}

fn compare_tol(cpu: &[f16], coreml: &[f16], label: &str, tol_max: f32, tol_rel: f64) {
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
    assert!(max_abs < tol_max, "{label}: max_abs {max_abs} > tol {tol_max}");
    assert!(rel_l2 < tol_rel, "{label}: rel_l2 {rel_l2} > tol {tol_rel}");
}

// =========================================================================
// Softmax
// =========================================================================

/// Softmax on rank-3 [B, S, D] tensor along the last axis (D).
/// Mirrors the post-attention-scores softmax in transformer self-attention,
/// where `[B, H, S, S]` is a common shape (we use B=H=2, S=4 here).
#[test]
fn softmax_rank3_last_axis() -> Result<()> {
    const B: usize = 2;
    const S: usize = 4;
    const D: usize = 6;

    let mut model = TypedModel::default();
    let input = model.add_source("input", f16::fact([B, S, D]))?;
    let sm = Softmax {
        axes: tvec![2],
        quant_output_dt: None,
        kind: SoftmaxKind::Softmax(Default::default()),
    };
    model.wire_node("softmax", sm, &[input])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let softmax_after =
        coreml_model.nodes.iter().filter(|n| n.op_as::<Softmax>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected Softmax to be absorbed into 1 CoremlOp");
    assert_eq!(softmax_after, 0, "Softmax should have no CPU residual");

    let raw: Vec<f16> = fill_sin(B * S * D, 1.0);
    let input_t = Tensor::from_shape::<f16>(&[B, S, D], &raw)?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![input_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![input_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, "softmax_rank3_last_axis");

    // Rows sum to ~1 (softmax invariant).
    for row in 0..(B * S) {
        let s: f32 = (0..D).map(|i| k[row * D + i].to_f32()).sum();
        assert!((s - 1.0).abs() < 1e-2, "softmax row {row} sums to {s}, expected ~1");
    }
    Ok(())
}

// =========================================================================
// General MatMul (runtime × runtime, batched)
// =========================================================================

/// Batched runtime × runtime matmul: `[B, M, K] @ [B, K, N]`. This is the
/// `attn @ V` shape in transformer attention (B = batch×heads, M/N = seq,
/// K = head_dim).
#[test]
fn general_matmul_runtime_runtime_batched() -> Result<()> {
    const B: usize = 2;
    const M: usize = 3;
    const K: usize = 4;
    const N: usize = 5;

    let mut model = TypedModel::default();
    let a = model.add_source("a", f16::fact([B, M, K]))?;
    let b = model.add_source("b", f16::fact([B, K, N]))?;
    let es = EinSum::new("bmk,bkn->bmn".parse()?, DatumType::F16);
    model.wire_node("matmul", es, &[a, b])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected runtime×runtime matmul to be absorbed into 1 CoremlOp");

    let a_t = Tensor::from_shape::<f16>(&[B, M, K], &fill_sin(B * M * K, 0.0))?;
    let b_t = Tensor::from_shape::<f16>(&[B, K, N], &fill_sin(B * K * N, 7.0))?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![a_t.clone().into(), b_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![a_t.into(), b_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, "general_matmul_runtime_runtime_batched");
    Ok(())
}

/// `Q @ K.T` shape: `[B, M, K] @ [B, N, K]` (transpose on second operand).
/// Verifies the `transpose_y=true` path of general_matmul.
#[test]
fn general_matmul_runtime_runtime_transpose_y() -> Result<()> {
    const B: usize = 2;
    const M: usize = 3;
    const K: usize = 4;
    const N: usize = 5;

    let mut model = TypedModel::default();
    let a = model.add_source("a", f16::fact([B, M, K]))?;
    let b = model.add_source("b", f16::fact([B, N, K]))?;
    let es = EinSum::new("bmk,bnk->bmn".parse()?, DatumType::F16);
    model.wire_node("matmul", es, &[a, b])?;
    model.auto_outputs()?;

    let cpu_model = model.clone();
    let mut coreml_model = model;
    CoremlTransform::default().transform(&mut coreml_model)?;

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    assert_eq!(coreml_count, 1, "expected transpose-y matmul to be absorbed into 1 CoremlOp");

    let a_t = Tensor::from_shape::<f16>(&[B, M, K], &fill_sin(B * M * K, 0.0))?;
    let b_t = Tensor::from_shape::<f16>(&[B, N, K], &fill_sin(B * N * K, 11.0))?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![a_t.clone().into(), b_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![a_t.into(), b_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    compare(c, k, "general_matmul_runtime_runtime_transpose_y");
    Ok(())
}

// =========================================================================
// LayerNorm
// =========================================================================

/// Hand-build the LayerNorm chain that tract's `into_decluttered` produces.
/// Wires:
///   mean_sum = ReduceSum(x, axes=[D])
///   mean = mean_sum * (1/D)
///   d = x - mean
///   dd = d * d
///   var_sum = ReduceSum(dd, axes=[D])
///   var = var_sum * (1/D)
///   var_eps = var + eps
///   inv_std = Rsqrt(var_eps)
///   normalized = d * inv_std
///   scaled = normalized * gamma
///   y = scaled + beta
fn build_layer_norm_chain(b: usize, s: usize, d: usize) -> Result<TypedModel> {
    use tract_core::ops::element_wise::ElementWiseOp;
    use tract_core::ops::nn::{Reduce, Reducer};

    let mut model = TypedModel::default();
    let x = model.add_source("input", f16::fact([b, s, d]))?;

    let recip_n = f16::from_f32(1.0 / d as f32);
    let recip_n_t = Tensor::from_shape::<f16>(&[1, 1, 1], &[recip_n])?;
    let recip_n_outlet = model.add_const("recip_n", recip_n_t)?;

    let mean_sum =
        model.wire_node("mean_sum", Reduce { axes: tvec![2], reducer: Reducer::Sum }, &[x])?[0];
    let mean = model.wire_node("mean", math::mul(), &[mean_sum, recip_n_outlet])?[0];
    let d_node = model.wire_node("d", math::sub(), &[x, mean])?[0];

    let dd = model.wire_node("dd", math::mul(), &[d_node, d_node])?[0];
    let var_sum =
        model.wire_node("var_sum", Reduce { axes: tvec![2], reducer: Reducer::Sum }, &[dd])?[0];
    let var = model.wire_node("var", math::mul(), &[var_sum, recip_n_outlet])?[0];

    let eps_t = Tensor::from_shape::<f16>(&[1, 1, 1], &[f16::from_f32(1e-5)])?;
    let eps_outlet = model.add_const("eps", eps_t)?;
    let var_eps = model.wire_node("var_eps", math::add(), &[var, eps_outlet])?[0];

    let rsqrt_op = ElementWiseOp(Box::new(tract_core::ops::math::Rsqrt {}), None);
    let inv_std = model.wire_node("inv_std", rsqrt_op, &[var_eps])?[0];
    let normalized = model.wire_node("normalized", math::mul(), &[d_node, inv_std])?[0];

    let gamma_data: Vec<f16> = (0..d).map(|i| f16::from_f32(0.5 + (i as f32) * 0.1)).collect();
    let gamma_t = Tensor::from_shape::<f16>(&[1, 1, d], &gamma_data)?;
    let gamma_outlet = model.add_const("gamma", gamma_t)?;
    let scaled = model.wire_node("scaled", math::mul(), &[normalized, gamma_outlet])?[0];

    let beta_data: Vec<f16> = (0..d).map(|i| f16::from_f32(-0.1 + (i as f32) * 0.02)).collect();
    let beta_t = Tensor::from_shape::<f16>(&[1, 1, d], &beta_data)?;
    let beta_outlet = model.add_const("beta", beta_t)?;
    model.wire_node("y", math::add(), &[scaled, beta_outlet])?;

    model.auto_outputs()?;
    Ok(model)
}

#[test]
fn layer_norm_chain_folds_to_single_mil_op() -> Result<()> {
    const B: usize = 2;
    const S: usize = 4;
    const D: usize = 8;

    // CPU reference runs on the *un-decluttered* chain to avoid a tract
    // RmsNorm + F16 eval-path bug — the chain runs fine elementwise but
    // tract's lowered RmsNorm op asserts F32 eps. CoreML side declutters
    // (since that's what real ONNX models look like after import) and
    // expects the LayerNorm-fold detector to fire.
    let cpu_model = build_layer_norm_chain(B, S, D)?;
    let mut coreml_model = build_layer_norm_chain(B, S, D)?.into_decluttered()?;

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let layer_norm_pre_count = coreml_model
        .nodes
        .iter()
        .filter(|n| n.op_as::<tract_core::ops::nn::RmsNorm>().is_some())
        .count();
    let _ = (coreml_count, layer_norm_pre_count);
    CoremlTransform::default().transform(&mut coreml_model)?;

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let rms_residual = coreml_model
        .nodes
        .iter()
        .filter(|n| n.op_as::<tract_core::ops::nn::RmsNorm>().is_some())
        .count();
    assert_eq!(
        coreml_count, 1,
        "expected LayerNorm chain to fold into 1 CoremlOp (got {coreml_count})"
    );
    assert_eq!(rms_residual, 0, "RmsNorm should be absorbed");

    let raw: Vec<f16> = fill_sin(B * S * D, 3.0);
    let input_t = Tensor::from_shape::<f16>(&[B, S, D], &raw)?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![input_t.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![input_t.into()])?;
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    // LayerNorm involves variance + rsqrt + multiple multiplies, so the
    // F16 round-off accumulates more than for pointwise ops. ~1.5e-2 max
    // error is typical; rel_l2 stays in the low-1e-3 regime.
    compare_tol(c, k, "layer_norm_chain_folds_to_single_mil_op", 2e-2, 1e-2);
    Ok(())
}

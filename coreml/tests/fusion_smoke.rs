//! Smoke tests for the subgraph-fusion path.
//!
//! These exercise `CoremlTransform`'s ability to grow a subgraph past one op
//! by translating Add as well as Conv. A `Conv → Add` chain on a synthetic
//! tract IR should fuse into one `CoremlOp` (one MLPackage), not two.

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::math;
use tract_core::ops::nn::DataFormat;
use tract_core::transform::ModelTransform;

use tract_coreml::{CoremlOp, CoremlTransform};

const IN_C: usize = 4;
const OUT_C: usize = 8;
const H: usize = 8;
const W: usize = 8;
const K: usize = 3;

/// Build: input → Conv → Add(self, self) → output.
/// `Add(self, self)` doubles the Conv output, giving a single-input single-output
/// model with a Conv→Add chain that should fuse into one CoremlOp.
fn build_conv_add_model() -> Result<TypedModel> {
    let weight: Vec<f16> = (0..(OUT_C * IN_C * K * K))
        .map(|i| f16::from_f32(((i as f32) * 0.07).sin() * 0.3))
        .collect();
    let bias: Vec<f16> =
        (0..OUT_C).map(|i| f16::from_f32(((i as f32) * 0.13).cos() * 0.1)).collect();
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
    // Add the Conv output to itself — single Add node consuming a single
    // outlet (twice). Output = 2 × conv(input).
    let _add_out = model.wire_node("add_self", math::add(), &[conv_out, conv_out])?[0];
    model.auto_outputs()?;
    Ok(model)
}

fn make_input() -> Result<Tensor> {
    let data: Vec<f16> =
        (0..(IN_C * H * W)).map(|i| f16::from_f32(((i as f32) * 0.04).cos() * 0.5)).collect();
    Tensor::from_shape::<f16>(&[1, IN_C, H, W], &data)
}

#[test]
fn conv_add_fuses_into_one_coreml_op() -> Result<()> {
    let mut model = build_conv_add_model()?;

    let convs_before = model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    let adds_before = model.nodes.iter().filter(|n| n.op.name() == "Add").count();
    assert_eq!(convs_before, 1, "expected 1 Conv before transform");
    assert_eq!(adds_before, 1, "expected 1 Add before transform");

    CoremlTransform::default().transform(&mut model)?;

    let convs_after = model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    let adds_after = model.nodes.iter().filter(|n| n.op.name() == "Add").count();
    let coreml_ops = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();

    assert_eq!(convs_after, 0, "Conv should be absorbed into CoremlOp");
    assert_eq!(adds_after, 0, "Add should be absorbed into CoremlOp");
    assert_eq!(coreml_ops, 1, "expected exactly 1 CoremlOp (Conv+Add fused), got {coreml_ops}");
    Ok(())
}

#[test]
fn conv_add_runtime_matches_cpu_within_fp16_tol() -> Result<()> {
    let cpu_model = build_conv_add_model()?;
    let mut coreml_model = cpu_model.clone();
    CoremlTransform::default().transform(&mut coreml_model)?;

    let input = make_input()?;

    let cpu_out = cpu_model.into_runnable()?.run(tvec![input.clone().into()])?;
    let coreml_out = coreml_model.into_runnable()?.run(tvec![input.into()])?;

    assert_eq!(cpu_out.len(), 1);
    assert_eq!(coreml_out.len(), 1);
    let c = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    assert_eq!(c.len(), k.len());

    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (cf, kf) in c.iter().zip(k.iter()) {
        let cv = cf.to_f32();
        let kv = kf.to_f32();
        let d = (cv - kv).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (cv as f64) * (cv as f64);
    }
    let rel_l2 = (sum_sq_diff / sum_sq_ref).sqrt();
    println!("[conv+add fused] max_abs_diff = {max_abs:.6}  rel_l2 = {rel_l2:.6e}");

    assert!(max_abs < 5e-3, "max_abs_diff {max_abs} exceeds FP16 tol");
    assert!(rel_l2 < 1e-2, "rel_l2 {rel_l2} exceeds FP16 tol");
    Ok(())
}

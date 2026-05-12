//! tract `SumPool` → MIL `avg_pool` (or `avg_pool` × area for pure-sum variant).
//!
//! tract decomposes ONNX `AveragePool` into a `SumPool` op with a
//! `normalize: bool` flag. When `normalize == true`, SumPool is exactly
//! AvgPool; we map directly to MIL `mb.avg_pool`. When `normalize == false`
//! (rare — used by some custom decompositions), we emit AvgPool followed
//! by a Mul-by-area constant.
//!
//! Phase 4 first cut: F16, NCHW or CHW, kernel rank 2, Explicit/Valid
//! padding. Same machinery as `maxpool.rs`.

use std::collections::HashMap;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::cnn::{PaddingSpec, SumPool};
use tract_core::ops::nn::DataFormat;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints, tv_strings};
use crate::proto::core_ml::specification::mil_spec as mil;
use crate::proto::core_ml::specification::mil_spec::tensor_value::RepeatedBytes;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum AvgPoolAnalysis {
    Translatable(AvgPoolPlan),
    Skip(String),
}

pub struct AvgPoolPlan {
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    pub kernel: Vec<i32>,
    pub strides: Vec<i32>,
    pub pad: Vec<i32>,
    pub pad_type: &'static str,
    /// True if the SumPool was the normalized (= AvgPool) variant. False if
    /// it was a pure sum and we need to multiply by `area = kH * kW`.
    pub normalized: bool,
    /// Whether to include padded zeros in the average denominator (Apple's
    /// `exclude_padding_from_average`). tract's `count_include_pad` is the
    /// inverse: count_include_pad → !exclude_padding_from_average.
    pub exclude_padding_from_average: bool,
    /// MIL `avg_pool` `ceil_mode`. Only meaningful when input/kernel/stride
    /// don't divide evenly — switches output-dim computation between
    /// `floor((D - K) / S) + 1` (false) and `ceil((D - K) / S) + 1` (true).
    /// RVM has a 2×2/stride-2 avg_pool on `45×60` input that needs
    /// ceil_mode=true to produce the `23×30` output the model expects.
    /// Without this, MIL/MPS computes `22×30` and concat with parallel
    /// branches fails: `mps.concat: input shapes must match except at axis`.
    pub ceil_mode: bool,
    pub output_fact: TypedFact,
}

pub fn analyse_sumpool(model: &TypedModel, node: &TypedNode) -> Result<AvgPoolAnalysis> {
    let Some(sp) = node.op_as::<SumPool>() else {
        return Ok(AvgPoolAnalysis::Skip("not a SumPool".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(AvgPoolAnalysis::Skip(format!(
            "SumPool node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }
    let prepend_n = match sp.pool_spec.data_format {
        DataFormat::NCHW => false,
        DataFormat::CHW => true,
        other => {
            return Ok(AvgPoolAnalysis::Skip(format!(
                "SumPool data_format={other:?} (need NCHW or CHW)"
            )));
        }
    };
    if sp.pool_spec.kernel_shape.len() != 2 {
        return Ok(AvgPoolAnalysis::Skip(format!(
            "SumPool kernel rank {} (need 2)",
            sp.pool_spec.kernel_shape.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(AvgPoolAnalysis::Skip(format!(
            "SumPool input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let expected_rank = if prepend_n { 3 } else { 4 };
    let raw_in = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) if s.len() == expected_rank => s,
        Some(s) => {
            return Ok(AvgPoolAnalysis::Skip(format!(
                "SumPool input rank {} (expected {} for {:?})",
                s.len(),
                expected_rank,
                sp.pool_spec.data_format
            )));
        }
        None => {
            return Ok(AvgPoolAnalysis::Skip(format!(
                "SumPool input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    let input_shape: Vec<i64> =
        if prepend_n { std::iter::once(1).chain(raw_in).collect() } else { raw_in };

    let out_fact = node.outputs[0].fact.clone();
    let raw_out = match shape_to_concrete_i64(&out_fact.shape) {
        Some(s) if s.len() == expected_rank => s,
        Some(s) => {
            return Ok(AvgPoolAnalysis::Skip(format!(
                "SumPool output rank {} (expected {} for {:?})",
                s.len(),
                expected_rank,
                sp.pool_spec.data_format
            )));
        }
        None => {
            return Ok(AvgPoolAnalysis::Skip(format!(
                "SumPool output symbolic shape: {:?}",
                out_fact.shape
            )));
        }
    };
    let output_shape: Vec<i64> =
        if prepend_n { std::iter::once(1).chain(raw_out).collect() } else { raw_out };

    let kernel: Vec<i32> = sp.pool_spec.kernel_shape.iter().map(|&k| k as i32).collect();
    let strides: Vec<i32> = match &sp.pool_spec.strides {
        Some(s) if s.len() == 2 => s.iter().map(|&v| v as i32).collect(),
        Some(s) => return Ok(AvgPoolAnalysis::Skip(format!("SumPool stride rank {}", s.len()))),
        None => vec![1, 1],
    };
    let (pad_type, pad, ceil_mode) = match &sp.pool_spec.padding {
        PaddingSpec::Valid => ("valid", vec![], false),
        // MIL `avg_pool` `pad_type="same"` matches ONNX SAME_UPPER semantics
        // (extra pad goes after the spatial dim when total pad is odd —
        // typical for stride>1). InceptionV3 surfaces SameUpper for every
        // intra-Inception-module 3×3 average pool. SameLower has no direct
        // MIL equivalent and is rarer; pre-compute and pass as Explicit if
        // we ever hit it.
        PaddingSpec::SameUpper => ("same", vec![], false),
        PaddingSpec::Explicit(b, a) => {
            if b.len() != 2 || a.len() != 2 {
                return Ok(AvgPoolAnalysis::Skip("SumPool Explicit pad rank".into()));
            }
            ("custom", vec![b[0] as i32, a[0] as i32, b[1] as i32, a[1] as i32], false)
        }
        // ExplicitOnnxPool carries `ceil_mode`, which controls whether
        // partial windows at the spatial-dim boundary contribute an extra
        // output position. RVM's 2×2/stride-2 avg_pool over [1,3,45,60]
        // with ceil_mode=true → output [1,3,23,30] (matches sibling
        // branches at the concat); with ceil_mode=false → [1,3,22,30] and
        // MPS rejects the concat. So we propagate ceil_mode rather than
        // forcing false.
        PaddingSpec::ExplicitOnnxPool(b, a, ceil_mode) => {
            if b.len() != 2 || a.len() != 2 {
                return Ok(AvgPoolAnalysis::Skip("SumPool Explicit pad rank".into()));
            }
            ("custom", vec![b[0] as i32, a[0] as i32, b[1] as i32, a[1] as i32], *ceil_mode)
        }
        other => return Ok(AvgPoolAnalysis::Skip(format!("SumPool padding {other:?}"))),
    };

    Ok(AvgPoolAnalysis::Translatable(AvgPoolPlan {
        input_shape,
        output_shape,
        kernel,
        strides,
        pad,
        pad_type,
        normalized: sp.normalize,
        exclude_padding_from_average: !sp.count_include_pad,
        ceil_mode,
        output_fact: out_fact,
    }))
}

pub fn emit_sumpool_mil(
    plan: &AvgPoolPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;
    let i32t_2 = tensor_type(DataType::Int32, &[2]);
    let i32t_4 = tensor_type(DataType::Int32, &[4]);
    let str_t = tensor_type(DataType::String, &[]);
    let bool_t = tensor_type(DataType::Bool, &[]);

    let kernel_n = format!("{p}_kernel");
    let strides_n = format!("{p}_strides");
    let pad_type_n = format!("{p}_pad_type");
    let ceil_n = format!("{p}_ceil_mode");
    let exclude_n = format!("{p}_exclude_pad");

    let kernel_op = op_const_immediate(&kernel_n, i32t_2.clone(), tv_ints(plan.kernel.clone()));
    let strides_op = op_const_immediate(&strides_n, i32t_2, tv_ints(plan.strides.clone()));
    let pad_type_op =
        op_const_immediate(&pad_type_n, str_t, tv_strings(vec![plan.pad_type.into()]));
    let ceil_val = mil::TensorValue {
        value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
            values: vec![plan.ceil_mode],
        })),
    };
    let ceil_op = op_const_immediate(&ceil_n, bool_t.clone(), ceil_val);
    let exclude_val = mil::TensorValue {
        value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
            values: vec![plan.exclude_padding_from_average],
        })),
    };
    let exclude_op = op_const_immediate(&exclude_n, bool_t, exclude_val);

    // MIL `avg_pool` requires `pad` ALWAYS (even when pad_type="valid"),
    // unlike `max_pool` which only requires it for pad_type="custom".
    // Emit zeros for valid.
    let pad_n = format!("{p}_pad");
    let pad_values: Vec<i32> =
        if plan.pad_type == "custom" { plan.pad.clone() } else { vec![0, 0, 0, 0] };
    let pad_op = op_const_immediate(&pad_n, i32t_4, tv_ints(pad_values));

    let mut prelude = vec![kernel_op, strides_op, pad_type_op, pad_op, ceil_op, exclude_op];

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("kernel_sizes".into(), arg_name(&kernel_n));
    inputs.insert("strides".into(), arg_name(&strides_n));
    inputs.insert("pad_type".into(), arg_name(&pad_type_n));
    inputs.insert("pad".into(), arg_name(&pad_n));
    inputs.insert("ceil_mode".into(), arg_name(&ceil_n));
    inputs.insert("exclude_padding_from_average".into(), arg_name(&exclude_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    // For the normalized case (= AvgPool), `mb.avg_pool` is the direct answer.
    // For the pure-sum case, emit avg_pool then multiply by area = kH * kW
    // (because tract's SumPool computes ∑ but mb.avg_pool computes ∑/kH·kW).
    let avgpool_name = if plan.normalized { output_name.to_string() } else { format!("{p}_avg") };
    let avg_op = mil::Operation {
        r#type: "avg_pool".into(),
        inputs,
        outputs: vec![mil::NamedValueType {
            name: avgpool_name.clone(),
            r#type: Some(out_ty.clone()),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(avg_op);

    if !plan.normalized {
        // Multiply by kH*kW to recover the pure sum.
        let area = (plan.kernel[0] * plan.kernel[1]) as f32;
        let area_n = format!("{p}_area");
        let area_f16 = f16::from_f32(area);
        let area_bytes = area_f16.to_le_bytes().to_vec();
        let area_op = op_const_immediate(
            &area_n,
            tensor_type(DataType::Float16, &[]),
            mil::TensorValue {
                value: Some(mil::tensor_value::Value::Bytes(RepeatedBytes { values: area_bytes })),
            },
        );
        prelude.push(area_op);
        let mul_inputs: HashMap<String, mil::Argument> =
            HashMap::from([("x".into(), arg_name(&avgpool_name)), ("y".into(), arg_name(&area_n))]);
        let mul_op = mil::Operation {
            r#type: "mul".into(),
            inputs: mul_inputs,
            outputs: vec![mil::NamedValueType {
                name: output_name.to_string(),
                r#type: Some(out_ty),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        };
        prelude.push(mul_op);
    }

    Ok(prelude)
}

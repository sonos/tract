//! tract `MaxPool` → MIL `max_pool` translator.
//!
//! 2D max pooling. SqueezeNet uses three of these between the early Conv +
//! fire-module blocks; without translation each splits the graph into a new
//! subgraph. Same `PoolSpec` machinery as Conv, so we reuse the data-format /
//! kernel-shape / stride / padding parsing patterns.
//!
//! Phase 3 first cut: F16, NCHW or CHW, kernel rank 2, no `with_index_outputs`.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::cnn::{MaxPool, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints, tv_strings};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum MaxPoolAnalysis {
    Translatable(MaxPoolPlan),
    Skip(String),
}

pub struct MaxPoolPlan {
    /// MLPackage-side input shape (rank 4).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (rank 4).
    pub output_shape: Vec<i64>,
    /// `[kH, kW]`.
    pub kernel: Vec<i32>,
    /// `[sH, sW]`.
    pub strides: Vec<i32>,
    /// `[pad_top, pad_bottom, pad_left, pad_right]`. Empty if pad_type != custom.
    pub pad: Vec<i32>,
    /// MIL `pad_type`: "valid" or "custom" or "same".
    pub pad_type: &'static str,
    /// MIL `ceil_mode`. Propagated from `PaddingSpec::ExplicitOnnxPool`'s
    /// third field (defaults to false otherwise). Same RVM-class concern as
    /// avgpool: a partial-window boundary needs ceil_mode=true to keep the
    /// MIL-side output shape matching tract's predicted shape.
    pub ceil_mode: bool,
    /// tract-side output fact (CoremlOp reshapes between this and rank-4).
    pub output_fact: TypedFact,
}

pub fn analyse_maxpool(model: &TypedModel, node: &TypedNode) -> Result<MaxPoolAnalysis> {
    let Some(mp) = node.op_as::<MaxPool>() else {
        return Ok(MaxPoolAnalysis::Skip("not a MaxPool".into()));
    };
    if mp.with_index_outputs.is_some() {
        return Ok(MaxPoolAnalysis::Skip("MaxPool with_index_outputs not supported".into()));
    }
    if node.inputs.len() != 1 {
        return Ok(MaxPoolAnalysis::Skip(format!(
            "MaxPool node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let prepend_n = match mp.pool_spec.data_format {
        DataFormat::NCHW => false,
        DataFormat::CHW => true,
        other => {
            return Ok(MaxPoolAnalysis::Skip(format!(
                "MaxPool data_format={other:?} (need NCHW or CHW)"
            )));
        }
    };
    if mp.pool_spec.kernel_shape.len() != 2 {
        return Ok(MaxPoolAnalysis::Skip(format!(
            "MaxPool kernel rank {} (need 2)",
            mp.pool_spec.kernel_shape.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(MaxPoolAnalysis::Skip(format!(
            "MaxPool input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let expected_rank = if prepend_n { 3 } else { 4 };
    let raw_in = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) if s.len() == expected_rank => s,
        Some(s) => {
            return Ok(MaxPoolAnalysis::Skip(format!(
                "MaxPool input rank {} (expected {} for {:?})",
                s.len(),
                expected_rank,
                mp.pool_spec.data_format
            )));
        }
        None => {
            return Ok(MaxPoolAnalysis::Skip(format!(
                "MaxPool input symbolic shape: {:?}",
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
            return Ok(MaxPoolAnalysis::Skip(format!(
                "MaxPool output rank {} (expected {} for {:?})",
                s.len(),
                expected_rank,
                mp.pool_spec.data_format
            )));
        }
        None => {
            return Ok(MaxPoolAnalysis::Skip(format!(
                "MaxPool output symbolic shape: {:?}",
                out_fact.shape
            )));
        }
    };
    let output_shape: Vec<i64> =
        if prepend_n { std::iter::once(1).chain(raw_out).collect() } else { raw_out };

    let kernel: Vec<i32> = mp.pool_spec.kernel_shape.iter().map(|&k| k as i32).collect();
    let strides = pool_strides(&mp.pool_spec)?;
    let (pad_type, pad, ceil_mode) = pool_padding(&mp.pool_spec)?;

    Ok(MaxPoolAnalysis::Translatable(MaxPoolPlan {
        input_shape,
        output_shape,
        kernel,
        strides,
        pad,
        pad_type,
        ceil_mode,
        output_fact: out_fact,
    }))
}

fn pool_strides(spec: &PoolSpec) -> Result<Vec<i32>> {
    Ok(match &spec.strides {
        Some(s) if s.len() == 2 => s.iter().map(|&v| v as i32).collect(),
        Some(s) => anyhow::bail!("pool stride rank {} (need 2)", s.len()),
        None => vec![1, 1],
    })
}

fn pool_padding(spec: &PoolSpec) -> Result<(&'static str, Vec<i32>, bool)> {
    Ok(match &spec.padding {
        PaddingSpec::Valid => ("valid", vec![], false),
        // MIL `max_pool` `pad_type="same"` matches ONNX SAME_UPPER semantics
        // (extra pad after, when total pad is odd). SameLower is rarer and
        // has no direct MIL equivalent; surface as a clean bail if hit.
        PaddingSpec::SameUpper => ("same", vec![], false),
        PaddingSpec::Explicit(before, after) => {
            if before.len() != 2 || after.len() != 2 {
                anyhow::bail!(
                    "pool Explicit padding rank ({}, {}) (need (2, 2))",
                    before.len(),
                    after.len()
                );
            }
            (
                "custom",
                vec![before[0] as i32, after[0] as i32, before[1] as i32, after[1] as i32],
                false,
            )
        }
        // ExplicitOnnxPool's third field is `ceil_mode`. Same reason as in
        // `avgpool::analyse_sumpool` for propagating it: the MIL-side output
        // dim has to match tract's predicted shape so downstream concats
        // pass MPS validation.
        PaddingSpec::ExplicitOnnxPool(before, after, ceil_mode) => {
            if before.len() != 2 || after.len() != 2 {
                anyhow::bail!(
                    "pool ExplicitOnnxPool padding rank ({}, {}) (need (2, 2))",
                    before.len(),
                    after.len()
                );
            }
            (
                "custom",
                vec![before[0] as i32, after[0] as i32, before[1] as i32, after[1] as i32],
                *ceil_mode,
            )
        }
        other => {
            anyhow::bail!("pool padding {:?} (only Explicit/Valid/SameUpper supported)", other)
        }
    })
}

pub fn emit_maxpool_mil(
    plan: &MaxPoolPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let i32t_2 = tensor_type(DataType::Int32, &[2]);
    let i32t_4 = tensor_type(DataType::Int32, &[4]);
    let str_t = tensor_type(DataType::String, &[]);
    let bool_t = tensor_type(DataType::Bool, &[]);

    let kernel_n = format!("{output_name}_kernel");
    let strides_n = format!("{output_name}_strides");
    let pad_type_n = format!("{output_name}_pad_type");
    let ceil_n = format!("{output_name}_ceil_mode");

    let kernel_op = op_const_immediate(&kernel_n, i32t_2.clone(), tv_ints(plan.kernel.clone()));
    let strides_op = op_const_immediate(&strides_n, i32t_2.clone(), tv_ints(plan.strides.clone()));
    let pad_type_op =
        op_const_immediate(&pad_type_n, str_t, tv_strings(vec![plan.pad_type.into()]));
    let ceil_op = op_const_immediate(
        &ceil_n,
        bool_t,
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: vec![plan.ceil_mode],
            })),
        },
    );

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("kernel_sizes".into(), arg_name(&kernel_n));
    inputs.insert("strides".into(), arg_name(&strides_n));
    inputs.insert("pad_type".into(), arg_name(&pad_type_n));
    inputs.insert("ceil_mode".into(), arg_name(&ceil_n));

    let mut prelude = vec![kernel_op, strides_op, pad_type_op, ceil_op];

    // MIL `max_pool` (like `avg_pool`) requires the `pad` parameter to be
    // present even when `pad_type=="valid"` — empty pad vec for valid emits
    // the right semantics. Same lesson as `avgpool.rs`.
    let pad_n = format!("{output_name}_pad");
    let pad_values: Vec<i32> =
        if plan.pad_type == "custom" { plan.pad.clone() } else { vec![0; 4] };
    let pad_op = op_const_immediate(&pad_n, i32t_4, tv_ints(pad_values));
    inputs.insert("pad".into(), arg_name(&pad_n));
    prelude.push(pad_op);

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let pool_op = mil::Operation {
        r#type: "max_pool".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(pool_op);
    Ok(prelude)
}

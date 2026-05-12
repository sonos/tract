//! tract `Cast` → MIL `cast` translator.
//!
//! tract's `f32_to_f16` transform wraps F32 ops in `Cast(F32→F16)` and
//! `Cast(F16→F32)` pairs to interface F32-only and F16-only nodes. Without a
//! Cast translator, every Cast breaks the fusion chain — Conv → Cast(F16→F32)
//! → ... can't merge. With this translator, the Casts stay inside the MLPackage
//! and Apple's compiler can fold whole chains together.
//!
//! Phase 2 first cut: handles the dtypes our other op translators currently
//! accept (F16 + F32 + I32). If a Cast targets a dtype we can't represent in
//! `MLMultiArrayDataType`, we Skip and the Cast stays on CPU (still correct,
//! just not fused).

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::cast::Cast;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_strings};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum CastAnalysis {
    Translatable(CastPlan),
    Skip(String),
}

pub struct CastPlan {
    /// Input shape (rank 4, normalised by prepending 1s if needed).
    pub input_shape: Vec<i64>,
    /// Source dtype (the input's tract DatumType).
    pub from_dt: DatumType,
    /// Target dtype (`cast.to`).
    pub to_dt: DatumType,
    /// tract-side output fact (may be rank 3 if CHW; CoremlOp reshapes between).
    pub output_fact: TypedFact,
}

pub fn analyse_cast(model: &TypedModel, node: &TypedNode) -> Result<CastAnalysis> {
    let Some(cast) = node.op_as::<Cast>() else {
        return Ok(CastAnalysis::Skip("not a Cast".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(CastAnalysis::Skip(format!(
            "Cast node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let input_fact = model.outlet_fact(node.inputs[0])?;
    let from_dt = input_fact.datum_type;
    let to_dt = cast.to;

    if !is_supported_dtype(from_dt) {
        return Ok(CastAnalysis::Skip(format!("Cast input dtype {from_dt:?} not supported")));
    }
    if !is_supported_dtype(to_dt) {
        return Ok(CastAnalysis::Skip(format!("Cast target dtype {to_dt:?} not supported")));
    }

    let raw_shape = match shape_to_concrete_i64(&input_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(CastAnalysis::Skip(format!(
                "Cast input symbolic shape: {:?}",
                input_fact.shape
            )));
        }
    };
    if raw_shape.len() > 4 {
        return Ok(CastAnalysis::Skip(format!(
            "Cast input rank {} (only ≤4 supported)",
            raw_shape.len()
        )));
    }
    let input_shape = super::rank::pad_to_rank_4(&raw_shape);

    Ok(CastAnalysis::Translatable(CastPlan {
        input_shape,
        from_dt,
        to_dt,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

fn is_supported_dtype(dt: DatumType) -> bool {
    matches!(dt, DatumType::F16 | DatumType::F32 | DatumType::I32 | DatumType::I8)
}

/// Map tract `DatumType` to the MIL `mb.cast(dtype="...")` string. Must match
/// `crate::tensor::datum_type_to_ml` for the MLMultiArray side.
fn mil_dtype_string(dt: DatumType) -> Option<&'static str> {
    Some(match dt {
        DatumType::F16 => "fp16",
        DatumType::F32 => "fp32",
        DatumType::I32 => "int32",
        DatumType::I8 => "int8",
        _ => return None,
    })
}

/// Map tract `DatumType` to the MIL `DataType` enum used in the output's
/// `tensor_type`.
fn mil_data_type(dt: DatumType) -> Option<DataType> {
    Some(match dt {
        DatumType::F16 => DataType::Float16,
        DatumType::F32 => DataType::Float32,
        DatumType::I32 => DataType::Int32,
        DatumType::I8 => DataType::Int8,
        _ => return None,
    })
}

pub fn emit_cast_mil(
    plan: &CastPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let target_str = mil_dtype_string(plan.to_dt)
        .ok_or_else(|| anyhow::anyhow!("emit_cast_mil: unsupported dtype {:?}", plan.to_dt))?;
    let target_dt = mil_data_type(plan.to_dt)
        .ok_or_else(|| anyhow::anyhow!("emit_cast_mil: unsupported MIL dtype {:?}", plan.to_dt))?;

    // Inline-const "dtype" parameter: MIL `cast` takes the target dtype as a
    // string-typed const argument. Prefix with output_name for uniqueness
    // across multiple casts in the same subgraph.
    let dtype_const_name = format!("{output_name}_dtype");
    let dtype_const = op_const_immediate(
        &dtype_const_name,
        tensor_type_scalar(DataType::String),
        tv_strings(vec![target_str.into()]),
    );

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("dtype".into(), arg_name(&dtype_const_name));

    // Output shape is identical to input shape (cast is element-wise).
    let out_ty = tensor_type(target_dt, &plan.input_shape);

    let cast_op = mil::Operation {
        r#type: "cast".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![dtype_const, cast_op])
}

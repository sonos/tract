//! tract `Pad` → MIL `pad` translator.
//!
//! Pads each axis with `(before, after)` zero/constant/reflect/edge values.
//! Mostly relevant for transformer models that pre-pad feature maps before
//! windowed attention so the spatial dim is divisible by the window size.
//! Hiera (SAM 2) uses Pad after the patch_embed conv.
//!
//! Mode mapping:
//!   * `Constant(0)` (and Constant(any-numeric)) → MIL `pad(mode="constant",
//!     constant_val=N)`. We currently only emit for value-0 const pad to
//!     avoid having to thread the constant tensor through.
//!   * `Reflect` → MIL `pad(mode="reflect")`.
//!   * `Edge` → MIL `pad(mode="replicate")`.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::array::{Pad, PadMode};

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints, tv_strings};
use crate::proto::core_ml::specification::mil_spec as mil;
use crate::proto::core_ml::specification::mil_spec::tensor_value::RepeatedBytes;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum PadAnalysis {
    Translatable(PadPlan),
    Skip(String),
}

pub struct PadPlan {
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    /// Flattened pad vector: `[before_0, after_0, before_1, after_1, ...]`.
    pub pad: Vec<i32>,
    /// MIL pad mode: "constant", "reflect", "replicate".
    pub mode: &'static str,
    /// For constant mode: the value to pad with (F32 in MIL).
    pub constant_val: f32,
    pub output_fact: TypedFact,
}

pub fn analyse_pad(model: &TypedModel, node: &TypedNode) -> Result<PadAnalysis> {
    let Some(p) = node.op_as::<Pad>() else {
        return Ok(PadAnalysis::Skip("not a Pad".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(PadAnalysis::Skip(format!(
            "Pad node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(PadAnalysis::Skip(format!(
            "Pad input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(PadAnalysis::Skip(format!("Pad input symbolic shape: {:?}", in_fact.shape)));
        }
    };
    if input_shape.len() != p.pads.len() {
        return Ok(PadAnalysis::Skip(format!(
            "Pad pads.len() {} != input rank {}",
            p.pads.len(),
            input_shape.len()
        )));
    }
    if !(2..=5).contains(&input_shape.len()) {
        return Ok(PadAnalysis::Skip(format!(
            "Pad input rank {} (only 2..=5 supported)",
            input_shape.len()
        )));
    }

    let (mode, constant_val) = match &p.mode {
        PadMode::Constant(t) => {
            // Read scalar constant. tract stores it as a 0-d tensor.
            let v = match t.datum_type() {
                DatumType::F16 => unsafe { t.as_slice_unchecked::<half::f16>()[0].to_f32() },
                DatumType::F32 => unsafe { t.as_slice_unchecked::<f32>()[0] },
                DatumType::F64 => unsafe { t.as_slice_unchecked::<f64>()[0] as f32 },
                other => {
                    return Ok(PadAnalysis::Skip(format!(
                        "Pad Constant has unsupported dtype {other:?}"
                    )));
                }
            };
            ("constant", v)
        }
        PadMode::Reflect => ("reflect", 0.0),
        PadMode::Edge => ("replicate", 0.0),
    };

    let output_shape: Vec<i64> = input_shape
        .iter()
        .zip(p.pads.iter())
        .map(|(&d, &(a, b))| d + a as i64 + b as i64)
        .collect();

    // Flatten pads to [before_0, after_0, before_1, after_1, ...].
    let pad: Vec<i32> = p.pads.iter().flat_map(|&(a, b)| [a as i32, b as i32]).collect();

    Ok(PadAnalysis::Translatable(PadPlan {
        input_shape,
        output_shape,
        pad,
        mode,
        constant_val,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_pad_mil(
    plan: &PadPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;
    let mut ops: Vec<mil::Operation> = Vec::new();

    let pad_n = format!("{p}_pad");
    let pad_ty = tensor_type(DataType::Int32, &[plan.pad.len() as i64]);
    ops.push(op_const_immediate(&pad_n, pad_ty, tv_ints(plan.pad.clone())));

    let mode_n = format!("{p}_mode");
    ops.push(op_const_immediate(
        &mode_n,
        tensor_type_scalar(DataType::String),
        tv_strings(vec![plan.mode.into()]),
    ));

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("pad".into(), arg_name(&pad_n));
    inputs.insert("mode".into(), arg_name(&mode_n));

    // MIL `pad` requires `constant_val` to be present for ALL modes
    // (constant / reflect / replicate) — the param is required by the op
    // signature even when reflect/replicate ignore the value. Emit it
    // unconditionally as an F16 scalar matching the input dtype.
    let cv_n = format!("{p}_constant_val");
    let cv_f16 = half::f16::from_f32(plan.constant_val);
    ops.push(op_const_immediate(
        &cv_n,
        tensor_type_scalar(DataType::Float16),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bytes(RepeatedBytes {
                values: cv_f16.to_le_bytes().to_vec(),
            })),
        },
    ));
    inputs.insert("constant_val".into(), arg_name(&cv_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let pad_op = mil::Operation {
        r#type: "pad".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    ops.push(pad_op);
    Ok(ops)
}

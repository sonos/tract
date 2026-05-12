//! tract `AxisOp::Rm(axis)` → MIL `squeeze` translator.
//!
//! `Rm(axis)` removes a single axis whose size is 1. SqueezeNet has two of
//! these at the classifier head (squeezing the H=1, W=1 axes left by the
//! global-average-pool reduction); MobileNet folds these elsewhere so we
//! haven't needed it before.
//!
//! Phase 3 first cut: F16 only; rank stays as-is on both tract and MIL sides
//! (no padding to rank 4) because Squeeze changes rank — the natural rank of
//! the predecessor's output is what flows in.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::change_axes::AxisOp;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum RmAxisAnalysis {
    Translatable(RmAxisPlan),
    Skip(String),
}

pub struct RmAxisPlan {
    /// MLPackage-side input shape (rank N).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (rank N-1; axis removed).
    pub output_shape: Vec<i64>,
    /// Axis to squeeze (in input rank coords).
    pub axis: i32,
    pub output_fact: TypedFact,
}

pub fn analyse_rm_axis(model: &TypedModel, node: &TypedNode) -> Result<RmAxisAnalysis> {
    let Some(ax) = node.op_as::<AxisOp>() else {
        return Ok(RmAxisAnalysis::Skip("not an AxisOp".into()));
    };
    let axis = match ax {
        AxisOp::Rm(a) => *a,
        AxisOp::Add(_) => {
            return Ok(RmAxisAnalysis::Skip("AxisOp::Add not handled here".into()));
        }
        other => {
            return Ok(RmAxisAnalysis::Skip(format!("AxisOp::{other:?} not supported (only Rm)")));
        }
    };

    if node.inputs.len() != 1 {
        return Ok(RmAxisAnalysis::Skip(format!(
            "RmAxis node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(RmAxisAnalysis::Skip(format!(
            "RmAxis input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(RmAxisAnalysis::Skip(format!(
                "RmAxis input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if axis >= input_shape.len() {
        return Ok(RmAxisAnalysis::Skip(format!(
            "RmAxis axis {axis} >= input rank {}",
            input_shape.len()
        )));
    }
    if input_shape[axis] != 1 {
        return Ok(RmAxisAnalysis::Skip(format!(
            "RmAxis axis {axis} has size {} (must be 1)",
            input_shape[axis]
        )));
    }
    if input_shape.len() <= 1 {
        return Ok(RmAxisAnalysis::Skip(
            "RmAxis would produce rank 0 (not supported by MIL squeeze)".into(),
        ));
    }

    let mut output_shape = input_shape.clone();
    output_shape.remove(axis);

    Ok(RmAxisAnalysis::Translatable(RmAxisPlan {
        input_shape,
        output_shape,
        axis: axis as i32,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_rm_axis_mil(
    plan: &RmAxisPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let i32t_1 = tensor_type(DataType::Int32, &[1]);
    let axes_n = format!("{output_name}_axes");
    let axes_op = op_const_immediate(&axes_n, i32t_1, tv_ints(vec![plan.axis]));

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("axes".into(), arg_name(&axes_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let sq_op = mil::Operation {
        r#type: "squeeze".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![axes_op, sq_op])
}

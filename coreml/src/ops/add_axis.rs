//! tract `AxisOp::Add(axis)` → MIL `expand_dims` translator.
//!
//! Adds a new axis of size 1 at the given position. Natural complement to
//! [`super::rm_axis`] (Squeeze). Common in transformer-class models that
//! Reshape via Add/Rm pairs around attention.
//!
//! Phase 3 first cut: F16 only; rank stays as-is on both tract and MIL sides
//! (no padding to rank 4) because AddAxis changes rank — the natural rank of
//! the predecessor's output flows in, +1 flows out.

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
pub enum AddAxisAnalysis {
    Translatable(AddAxisPlan),
    Skip(String),
}

pub struct AddAxisPlan {
    /// MLPackage-side input shape (rank N).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (rank N+1; size-1 axis inserted).
    pub output_shape: Vec<i64>,
    /// Axis to insert (in output rank coords; 0..=input_rank).
    pub axis: i32,
    pub output_fact: TypedFact,
}

pub fn analyse_add_axis(model: &TypedModel, node: &TypedNode) -> Result<AddAxisAnalysis> {
    let Some(ax) = node.op_as::<AxisOp>() else {
        return Ok(AddAxisAnalysis::Skip("not an AxisOp".into()));
    };
    let axis = match ax {
        AxisOp::Add(a) => *a,
        AxisOp::Rm(_) => {
            return Ok(AddAxisAnalysis::Skip("AxisOp::Rm handled by rm_axis".into()));
        }
        other => {
            return Ok(AddAxisAnalysis::Skip(format!(
                "AxisOp::{other:?} not supported (only Add)"
            )));
        }
    };

    if node.inputs.len() != 1 {
        return Ok(AddAxisAnalysis::Skip(format!(
            "AddAxis node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(AddAxisAnalysis::Skip(format!(
            "AddAxis input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(AddAxisAnalysis::Skip(format!(
                "AddAxis input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if axis > input_shape.len() {
        return Ok(AddAxisAnalysis::Skip(format!(
            "AddAxis axis {axis} > input rank {}",
            input_shape.len()
        )));
    }
    if input_shape.len() >= 5 {
        return Ok(AddAxisAnalysis::Skip(format!(
            "AddAxis output rank {} > 5 (MIL expand_dims supports rank ≤ 5)",
            input_shape.len() + 1
        )));
    }

    let mut output_shape = input_shape.clone();
    output_shape.insert(axis, 1);

    Ok(AddAxisAnalysis::Translatable(AddAxisPlan {
        input_shape,
        output_shape,
        axis: axis as i32,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_add_axis_mil(
    plan: &AddAxisPlan,
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
    let exp_op = mil::Operation {
        r#type: "expand_dims".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![axes_op, exp_op])
}

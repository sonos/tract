//! tract `AxisOp::Reshape(at, from, to)` → MIL `reshape` translator.
//!
//! Reshape changes a contiguous slice of axes from one shape `from` to
//! another `to` while keeping prefix axes [0, at) and suffix axes [at +
//! from.len(), rank) unchanged. The total element count must agree:
//! `prod(from) == prod(to)`. Maps cleanly to MIL `reshape(x, shape)` where
//! `shape` is the full target shape (prefix + to + suffix).
//!
//! Critical for transformer models: every windowed-attention block in
//! Hiera/Swin-class encoders has multiple Reshapes that partition the
//! feature map into windows and stitch them back. Without this translator
//! every Reshape is a CPU residual that fragments the subgraph at the
//! attention boundary.

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
pub enum ReshapeAnalysis {
    Translatable(ReshapePlan),
    Skip(String),
}

pub struct ReshapePlan {
    /// Tract-view input shape (may be rank > 5 with unit dims).
    pub input_shape: Vec<i64>,
    /// Tract-view output shape (may be rank > 5 with unit dims).
    pub output_shape: Vec<i64>,
    /// **MLPackage-boundary input shape**: `input_shape` with leading unit
    /// dims stripped if input rank > 5. CoremlOp does the squeeze at the
    /// tract-side ↔ MLPackage-side boundary. Equal to `input_shape` when
    /// input rank ≤ 5.
    pub input_external_shape: Vec<i64>,
    /// Same for output.
    pub output_external_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_reshape(model: &TypedModel, node: &TypedNode) -> Result<ReshapeAnalysis> {
    let Some(ax) = node.op_as::<AxisOp>() else {
        return Ok(ReshapeAnalysis::Skip("not an AxisOp".into()));
    };
    let (at, from, to) = match ax {
        AxisOp::Reshape(at, from, to) => (*at, from, to),
        AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Move(_, _) => {
            return Ok(ReshapeAnalysis::Skip(
                "AxisOp::Add/Rm/Move handled by their dedicated translators".into(),
            ));
        }
    };

    if node.inputs.len() != 1 {
        return Ok(ReshapeAnalysis::Skip(format!(
            "Reshape node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(ReshapeAnalysis::Skip(format!(
            "Reshape input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(ReshapeAnalysis::Skip(format!(
                "Reshape input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };

    // Resolve `from` and `to` to concrete dims.
    let from_concrete: Vec<i64> = match from.iter().map(|d| d.to_i64().ok()).collect::<Option<_>>()
    {
        Some(v) => v,
        None => {
            return Ok(ReshapeAnalysis::Skip(format!(
                "Reshape `from` has symbolic dims: {from:?}"
            )));
        }
    };
    let to_concrete: Vec<i64> = match to.iter().map(|d| d.to_i64().ok()).collect::<Option<_>>() {
        Some(v) => v,
        None => {
            return Ok(ReshapeAnalysis::Skip(format!("Reshape `to` has symbolic dims: {to:?}")));
        }
    };

    // Sanity: total element count must match.
    let from_prod: i64 = from_concrete.iter().product();
    let to_prod: i64 = to_concrete.iter().product();
    if from_prod != to_prod {
        return Ok(ReshapeAnalysis::Skip(format!(
            "Reshape from-product {from_prod} != to-product {to_prod}"
        )));
    }

    // Sanity: input_shape[at..at+from.len()] should match `from`.
    if at + from_concrete.len() > input_shape.len() {
        return Ok(ReshapeAnalysis::Skip(format!(
            "Reshape at={at} + from-len={} exceeds input rank {}",
            from_concrete.len(),
            input_shape.len()
        )));
    }
    for (i, &v) in from_concrete.iter().enumerate() {
        if input_shape[at + i] != v {
            return Ok(ReshapeAnalysis::Skip(format!(
                "Reshape `from`[{i}] = {v} doesn't match input_shape[{}] = {}",
                at + i,
                input_shape[at + i]
            )));
        }
    }

    // Output shape = input_shape[0..at] ++ to ++ input_shape[at+from.len()..]
    let mut output_shape: Vec<i64> =
        Vec::with_capacity(input_shape.len() - from_concrete.len() + to_concrete.len());
    output_shape.extend_from_slice(&input_shape[..at]);
    output_shape.extend_from_slice(&to_concrete);
    output_shape.extend_from_slice(&input_shape[at + from_concrete.len()..]);

    // Boundary-strip rank > 5 shapes if they have leading unit dims.
    // Tract's declutter sometimes inserts unit-dim "decorator" axes at
    // position 0 (via AxisOp::Add(0)) that push intermediate ranks above
    // MIL's rank-5 cap. CoremlOp does the squeeze/unsqueeze at the
    // tract-side ↔ MLPackage-side boundary.
    let input_external_shape = match super::rank::try_strip_to_rank5(&input_shape) {
        Some(s) => s,
        None => {
            return Ok(ReshapeAnalysis::Skip(format!(
                "Reshape input rank {} > 5 with no leading unit dims to strip",
                input_shape.len()
            )));
        }
    };
    let output_external_shape = match super::rank::try_strip_to_rank5(&output_shape) {
        Some(s) => s,
        None => {
            return Ok(ReshapeAnalysis::Skip(format!(
                "Reshape output rank {} > 5 with no leading unit dims to strip (MIL reshape supports rank ≤ 5)",
                output_shape.len()
            )));
        }
    };

    Ok(ReshapeAnalysis::Translatable(ReshapePlan {
        input_shape,
        output_shape,
        input_external_shape,
        output_external_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_reshape_mil(
    plan: &ReshapePlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    // Use the EXTERNAL shapes (post-strip if rank > 5). The MLPackage value
    // flowing in is at input_external_shape rank; we reshape to
    // output_external_shape rank. CoremlOp at the boundary handles the
    // tract-rank ↔ external-rank conversion via metadata-only reshape.
    let shape_n = format!("{output_name}_shape");
    let i32t = tensor_type(DataType::Int32, &[plan.output_external_shape.len() as i64]);
    let shape_op = op_const_immediate(
        &shape_n,
        i32t,
        tv_ints(plan.output_external_shape.iter().map(|&v| v as i32).collect()),
    );

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(input_name));
    inputs.insert("shape".into(), arg_name(&shape_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_external_shape);
    let reshape_op = mil::Operation {
        r#type: "reshape".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![shape_op, reshape_op])
}

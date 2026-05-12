//! tract `MultiBroadcastTo` (= Expand) → MIL `broadcast_to` translator.
//!
//! `MultiBroadcastTo` broadcasts an input tensor to a (statically-known)
//! target shape — semantically the same as ONNX `Expand`. Common in models
//! that produce per-channel scalars (e.g. per-channel scale via SE blocks)
//! and need to apply them across spatial dims. Surfaces in RVM 4× per
//! frame for the SE-block / decoder broadcasts.
//!
//! MIL has `mb.fill_like` and `mb.broadcast_to(x, shape)`; the latter is
//! the right fit. Phase 4 first cut: F16, target shape concrete (rank 2..=5).

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::array::MultiBroadcastTo;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum BroadcastAnalysis {
    Translatable(BroadcastPlan),
    Skip(String),
}

pub struct BroadcastPlan {
    /// MLPackage-side input shape.
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (the broadcast target).
    pub output_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_broadcast(model: &TypedModel, node: &TypedNode) -> Result<BroadcastAnalysis> {
    let Some(mb) = node.op_as::<MultiBroadcastTo>() else {
        return Ok(BroadcastAnalysis::Skip("not a MultiBroadcastTo".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(BroadcastAnalysis::Skip(format!(
            "MultiBroadcastTo node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(BroadcastAnalysis::Skip(format!(
            "MultiBroadcastTo input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(BroadcastAnalysis::Skip(format!(
                "MultiBroadcastTo input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    let output_shape = match shape_to_concrete_i64(&mb.shape) {
        Some(s) => s,
        None => {
            return Ok(BroadcastAnalysis::Skip(format!(
                "MultiBroadcastTo target symbolic shape: {:?}",
                mb.shape
            )));
        }
    };
    if !(2..=5).contains(&output_shape.len()) {
        return Ok(BroadcastAnalysis::Skip(format!(
            "MultiBroadcastTo output rank {} (only 2..=5 supported)",
            output_shape.len()
        )));
    }

    Ok(BroadcastAnalysis::Translatable(BroadcastPlan {
        input_shape,
        output_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_broadcast_mil(
    plan: &BroadcastPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;
    let reps_n = format!("{p}_reps");

    // MIL doesn't have a direct `broadcast_to(x, shape)` op (older versions
    // did under different names; the iOS17 opset removed it). `tile` with
    // `reps[i] = output_shape[i] / input_shape[i]` is the equivalent.
    // Padding the input rank with leading 1s (left-broadcast) when ranks
    // differ matches NumPy / ONNX broadcasting semantics.
    let in_rank = plan.input_shape.len();
    let out_rank = plan.output_shape.len();
    let pad = out_rank.saturating_sub(in_rank);
    let padded_in: Vec<i64> =
        std::iter::repeat_n(1i64, pad).chain(plan.input_shape.iter().copied()).collect();
    let reps: Vec<i32> =
        padded_in.iter().zip(plan.output_shape.iter()).map(|(&i, &o)| (o / i) as i32).collect();
    let reps_op = op_const_immediate(
        &reps_n,
        tensor_type(DataType::Int32, &[out_rank as i64]),
        tv_ints(reps),
    );

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let tile_op = mil::Operation {
        r#type: "tile".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(input_name)),
            ("reps".into(), arg_name(&reps_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![reps_op, tile_op])
}

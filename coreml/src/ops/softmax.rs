//! tract `Softmax` (and `LogSoftmax`) → MIL `softmax`/`log_softmax` translator.
//!
//! Direct one-to-one mapping. tract preserves Softmax as a first-class op
//! (not decomposed into exp + sum + div) because the standard numerically-
//! stable form `exp(x - max) / sum(exp(x - max))` involves a per-row max
//! that's awkward to express as separate ops without losing the safe-eval
//! contract. The MIL `softmax` op handles all of that internally and is on
//! ANE's hardware fastpath — critical for transformer attention blocks where
//! Softmax is the bottleneck after the QK matmul.
//!
//! tract's `Softmax` carries `axes: TVec<usize>` (one or more axes that
//! together form the "row" being normalised). MIL `softmax` takes a single
//! `axis: int32`. We require exactly one axis here — multi-axis softmax is
//! rare outside niche networks (and would need a reshape pre-pass that
//! doesn't surface in the canary models).
//!
//! Rank-padding correction: most upstream translators pad rank-3 CHW tensors
//! to rank-4 NCHW by prepending N=1, so the tract-side axis must be shifted
//! by `(4 - tract_rank)` to match the MLPackage tensor that actually flows
//! in. Same correction we apply in `reduce.rs`.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::nn::{Softmax, SoftmaxKind};

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum SoftmaxAnalysis {
    Translatable(SoftmaxPlan),
    Skip(String),
}

pub struct SoftmaxPlan {
    /// MLPackage-side input/output shape (rank-padded to 4 if tract-side rank < 4).
    pub input_shape: Vec<i64>,
    /// Single axis in MLPackage rank coords (after rank-padding shift).
    pub axis: i32,
    /// MIL op name: "softmax" or "log_softmax".
    pub mil_op: &'static str,
    pub output_fact: TypedFact,
}

pub fn analyse_softmax(model: &TypedModel, node: &TypedNode) -> Result<SoftmaxAnalysis> {
    let Some(sm) = node.op_as::<Softmax>() else {
        return Ok(SoftmaxAnalysis::Skip("not a Softmax".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(SoftmaxAnalysis::Skip(format!(
            "Softmax has {} inputs (need 1)",
            node.inputs.len()
        )));
    }
    if sm.quant_output_dt.is_some() {
        return Ok(SoftmaxAnalysis::Skip("quantized Softmax not supported".into()));
    }
    let mil_op = match sm.kind {
        SoftmaxKind::Softmax(_) => "softmax",
        SoftmaxKind::LogSoftmax => "log_softmax",
    };
    if sm.axes.len() != 1 {
        return Ok(SoftmaxAnalysis::Skip(format!(
            "Softmax with {} axes not supported (MIL takes single axis)",
            sm.axes.len()
        )));
    }
    let raw_axis = sm.axes[0];

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(SoftmaxAnalysis::Skip(format!(
            "Softmax input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let raw_in = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(SoftmaxAnalysis::Skip(format!(
                "Softmax input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if !(2..=5).contains(&raw_in.len()) {
        return Ok(SoftmaxAnalysis::Skip(format!(
            "Softmax input rank {} (only 2..=5 supported)",
            raw_in.len()
        )));
    }

    // Rank-padding correction: see reduce.rs for the full rationale.
    let pad = 4_usize.saturating_sub(raw_in.len());
    let input_shape = super::rank::pad_to_rank_4(&raw_in);
    let axis = raw_axis as i32 + pad as i32;
    if axis < 0 || (axis as usize) >= input_shape.len() {
        return Ok(SoftmaxAnalysis::Skip(format!(
            "Softmax axis {axis} out of range for rank {}",
            input_shape.len()
        )));
    }

    Ok(SoftmaxAnalysis::Translatable(SoftmaxPlan {
        input_shape,
        axis,
        mil_op,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_softmax_mil(
    plan: &SoftmaxPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let axis_n = format!("{output_name}_axis");
    let axis_op =
        op_const_immediate(&axis_n, tensor_type_scalar(DataType::Int32), tv_ints(vec![plan.axis]));

    let out_ty = tensor_type(DataType::Float16, &plan.input_shape);
    let sm_op = mil::Operation {
        r#type: plan.mil_op.into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(input_name)),
            ("axis".into(), arg_name(&axis_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![axis_op, sm_op])
}

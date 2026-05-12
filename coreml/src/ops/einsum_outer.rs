//! tract `EinSum` (no-K outer-product) → MIL `mb.mul` translator.
//!
//! Handles einsums where there's NO axis appearing in both inputs (no K
//! contraction). These aren't matmuls — they're pure element-wise products
//! with NumPy-style broadcast across the output dims.
//!
//! Example surfaces:
//! - **`m,an->abnm`** (RoPE in SmolLM2-class LLMs): `output[a,b,n,m] =
//!   A[m] * B[a,n]`. The `b` axis appears only in output (broadcast dim
//!   with size from the output fact, typically 1).
//!
//! Algorithm:
//! 1. Determine each axis's position in output, A, B.
//! 2. For input A, build a "target shape" of rank=output_rank with size-1
//!    for axes A doesn't have, A's actual size for axes A has.
//! 3. Same for B.
//! 4. mb.reshape(A) → A_padded (rank == output_rank).
//! 5. mb.reshape(B) → B_padded (rank == output_rank).
//! 6. mb.mul(A_padded, B_padded) — MIL's NumPy-broadcast handles the unit
//!    dims and any size-1→N broadcasts.
//!
//! Limitation: any output axis with size > 1 that appears in NEITHER input
//! AND ISN'T a broadcast (i.e. b=1) would need an explicit `mb.tile` after
//! the multiply; not yet handled. Raises a Skip when detected.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::axes::Axis;
use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum EinsumOuterAnalysis {
    Translatable(EinsumOuterPlan),
    Skip(String),
}

pub enum EinsumOuterOperand {
    Runtime { shape: Vec<i64> },
    Const { tensor: Tensor },
}

pub struct EinsumOuterPlan {
    pub a: EinsumOuterOperand,
    pub b: EinsumOuterOperand,
    /// Per-input rank-output target shape (size-1-padded for missing axes).
    pub a_padded_shape: Vec<i64>,
    pub b_padded_shape: Vec<i64>,
    /// Output shape (rank N).
    pub output_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_einsum_outer(model: &TypedModel, node: &TypedNode) -> Result<EinsumOuterAnalysis> {
    let Some(es) = node.op_as::<EinSum>() else {
        return Ok(EinsumOuterAnalysis::Skip("not an EinSum".into()));
    };
    if node.inputs.len() != 2 {
        return Ok(EinsumOuterAnalysis::Skip(format!(
            "EinSum has {} inputs (need 2)",
            node.inputs.len()
        )));
    }

    // Walk the AxesMapping. Bucket each axis by (in input 0, in input 1, in output).
    // We're looking for the K-less case: NO axis is in both inputs.
    let axes: Vec<&Axis> = es.axes.iter_all_axes().collect();
    for ax in &axes {
        let in0 = !ax.inputs[0].is_empty();
        let in1 = !ax.inputs[1].is_empty();
        if in0 && in1 {
            return Ok(EinsumOuterAnalysis::Skip(format!(
                "EinSum {} has K axis '{}' (this translator only handles no-K case)",
                es.axes, ax.repr
            )));
        }
    }
    // At this point we know K is empty.
    let rank_out = es.axes.rank(InOut::Out(0));
    let rank_a = es.axes.rank(InOut::In(0));
    let rank_b = es.axes.rank(InOut::In(1));

    // Resolve operands + shapes.
    let (a, a_shape) = resolve_operand(model, node.inputs[0])?;
    let (b, b_shape) = resolve_operand(model, node.inputs[1])?;
    if matches!((&a, &b), (EinsumOuterOperand::Const { .. }, EinsumOuterOperand::Const { .. })) {
        return Ok(EinsumOuterAnalysis::Skip(
            "EinSum has both inputs Const (should have been const-folded)".into(),
        ));
    }
    if a_shape.len() != rank_a || b_shape.len() != rank_b {
        return Ok(EinsumOuterAnalysis::Skip(format!(
            "EinSum {} shape rank mismatch with axes (a: shape={a_shape:?} axes_rank={rank_a}; \
             b: shape={b_shape:?} axes_rank={rank_b})",
            es.axes
        )));
    }

    // Compute the output shape from each axis's appearance.
    let mut output_shape = vec![0i64; rank_out];
    for ax in &axes {
        if ax.outputs[0].is_empty() {
            // K axis — but we already checked there are none. Skip defensively.
            continue;
        }
        if ax.outputs[0].len() != 1 {
            return Ok(EinsumOuterAnalysis::Skip(format!(
                "EinSum {} axis '{}' has {} positions in output (need 1)",
                es.axes,
                ax.repr,
                ax.outputs[0].len()
            )));
        }
        let out_pos = ax.outputs[0][0];
        // Get the dim from input A or input B (or default to 1 if neither —
        // pure-broadcast axis from output_fact).
        let dim = if !ax.inputs[0].is_empty() {
            if ax.inputs[0].len() != 1 {
                return Ok(EinsumOuterAnalysis::Skip(format!(
                    "EinSum {} axis '{}' multiple positions in A",
                    es.axes, ax.repr
                )));
            }
            a_shape[ax.inputs[0][0]]
        } else if !ax.inputs[1].is_empty() {
            if ax.inputs[1].len() != 1 {
                return Ok(EinsumOuterAnalysis::Skip(format!(
                    "EinSum {} axis '{}' multiple positions in B",
                    es.axes, ax.repr
                )));
            }
            b_shape[ax.inputs[1][0]]
        } else {
            // Pure expand-out axis — get size from the tract output fact.
            let out_fact = &node.outputs[0].fact;
            match shape_to_concrete_i64(&out_fact.shape) {
                Some(s) if out_pos < s.len() => s[out_pos],
                _ => {
                    return Ok(EinsumOuterAnalysis::Skip(format!(
                        "EinSum {} expand-out axis '{}' size unknown",
                        es.axes, ax.repr
                    )));
                }
            }
        };
        output_shape[out_pos] = dim;
    }

    // Build per-input padded shapes: rank == rank_out, with size-1 for axes
    // the input doesn't have, and the input's actual size at the input's
    // own axis position re-indexed to output position.
    let mut a_padded = vec![1i64; rank_out];
    let mut b_padded = vec![1i64; rank_out];
    for ax in &axes {
        if ax.outputs[0].is_empty() {
            continue;
        }
        let out_pos = ax.outputs[0][0];
        if !ax.inputs[0].is_empty() {
            a_padded[out_pos] = a_shape[ax.inputs[0][0]];
        }
        if !ax.inputs[1].is_empty() {
            b_padded[out_pos] = b_shape[ax.inputs[1][0]];
        }
    }

    // Sanity: the multiply-with-broadcast must produce output_shape. For each
    // dim, a_padded[i] and b_padded[i] must broadcast to output_shape[i].
    for i in 0..rank_out {
        let a_dim = a_padded[i];
        let b_dim = b_padded[i];
        let max = a_dim.max(b_dim);
        if (a_dim != 1 && a_dim != max) || (b_dim != 1 && b_dim != max) {
            return Ok(EinsumOuterAnalysis::Skip(format!(
                "EinSum {} dim {i} not broadcast-compatible: a={a_dim}, b={b_dim}",
                es.axes
            )));
        }
        if max != output_shape[i] {
            return Ok(EinsumOuterAnalysis::Skip(format!(
                "EinSum {} broadcast dim {i} = {max} doesn't match output {} \
                 (would need explicit tile)",
                es.axes, output_shape[i]
            )));
        }
    }

    Ok(EinsumOuterAnalysis::Translatable(EinsumOuterPlan {
        a,
        b,
        a_padded_shape: a_padded,
        b_padded_shape: b_padded,
        output_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

fn resolve_operand(model: &TypedModel, outlet: OutletId) -> Result<(EinsumOuterOperand, Vec<i64>)> {
    if let Some(t) = const_tensor(model, outlet)? {
        let shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
        return Ok((EinsumOuterOperand::Const { tensor: t.into_owned() }, shape));
    }
    let fact = model.outlet_fact(outlet)?;
    let shape = shape_to_concrete_i64(&fact.shape)
        .ok_or_else(|| anyhow::anyhow!("EinSum operand symbolic shape: {:?}", fact.shape))?;
    Ok((EinsumOuterOperand::Runtime { shape: shape.clone() }, shape))
}

pub fn emit_einsum_outer_mil(
    plan: &EinsumOuterPlan,
    blob: &mut BlobBuilder,
    a_data_name: Option<&str>,
    b_data_name: Option<&str>,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let mut prelude: Vec<mil::Operation> = Vec::new();

    // Resolve A: either pass through the runtime name, or write a const blob.
    let a_name = match (&plan.a, a_data_name) {
        (EinsumOuterOperand::Runtime { .. }, Some(n)) => n.to_string(),
        (EinsumOuterOperand::Runtime { .. }, None) => {
            anyhow::bail!("EinsumOuter::emit: a is Runtime but no name passed (fusion bug)")
        }
        (EinsumOuterOperand::Const { tensor }, None) => {
            let const_n = format!("{output_name}_a_const");
            let raw_shape: Vec<i64> = tensor.shape().iter().map(|&s| s as i64).collect();
            let const_ty = tensor_type(DataType::Float16, &raw_shape);
            let off = blob.add(BlobDataType::Float16, tensor.as_bytes());
            prelude.push(op_const_blob(&const_n, const_ty, mlpackage::WEIGHT_BLOB_PATH, off));
            const_n
        }
        (EinsumOuterOperand::Const { .. }, Some(_)) => {
            anyhow::bail!("EinsumOuter::emit: a is Const but caller passed a name (fusion bug)")
        }
    };
    let b_name = match (&plan.b, b_data_name) {
        (EinsumOuterOperand::Runtime { .. }, Some(n)) => n.to_string(),
        (EinsumOuterOperand::Runtime { .. }, None) => {
            anyhow::bail!("EinsumOuter::emit: b is Runtime but no name passed (fusion bug)")
        }
        (EinsumOuterOperand::Const { tensor }, None) => {
            let const_n = format!("{output_name}_b_const");
            let raw_shape: Vec<i64> = tensor.shape().iter().map(|&s| s as i64).collect();
            let const_ty = tensor_type(DataType::Float16, &raw_shape);
            let off = blob.add(BlobDataType::Float16, tensor.as_bytes());
            prelude.push(op_const_blob(&const_n, const_ty, mlpackage::WEIGHT_BLOB_PATH, off));
            const_n
        }
        (EinsumOuterOperand::Const { .. }, Some(_)) => {
            anyhow::bail!("EinsumOuter::emit: b is Const but caller passed a name (fusion bug)")
        }
    };

    // Reshape A to padded shape. Reshape B to padded shape. Multiply.
    let a_pad_n = format!("{output_name}_a_padded");
    let a_shape_n = format!("{output_name}_a_pad_shape");
    let a_shape_op = op_const_immediate(
        &a_shape_n,
        tensor_type(DataType::Int32, &[plan.a_padded_shape.len() as i64]),
        tv_ints(plan.a_padded_shape.iter().map(|&v| v as i32).collect()),
    );
    let a_pad_ty = tensor_type(DataType::Float16, &plan.a_padded_shape);
    prelude.push(a_shape_op);
    prelude.push(mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&a_name)),
            ("shape".into(), arg_name(&a_shape_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: a_pad_n.clone(), r#type: Some(a_pad_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    let b_pad_n = format!("{output_name}_b_padded");
    let b_shape_n = format!("{output_name}_b_pad_shape");
    let b_shape_op = op_const_immediate(
        &b_shape_n,
        tensor_type(DataType::Int32, &[plan.b_padded_shape.len() as i64]),
        tv_ints(plan.b_padded_shape.iter().map(|&v| v as i32).collect()),
    );
    let b_pad_ty = tensor_type(DataType::Float16, &plan.b_padded_shape);
    prelude.push(b_shape_op);
    prelude.push(mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&b_name)),
            ("shape".into(), arg_name(&b_shape_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: b_pad_n.clone(), r#type: Some(b_pad_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let mul_inputs: HashMap<String, mil::Argument> =
        HashMap::from([("x".into(), arg_name(&a_pad_n)), ("y".into(), arg_name(&b_pad_n))]);
    prelude.push(mil::Operation {
        r#type: "mul".into(),
        inputs: mul_inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    Ok(prelude)
}

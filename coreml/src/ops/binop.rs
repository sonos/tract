//! tract `TypedBinOp(Add | Mul | Max)` → MIL `add` / `mul` / `maximum` translator.
//!
//! Phase 2 capabilities:
//! - Both inputs F16
//! - Inputs may be either DATA (from another op's output, wired by name) or
//!   CONST (a `Const` or `Cast(Const)` chain — absorbed into the MLPackage
//!   weight blob and referenced via `op_const_blob`)
//! - NumPy-style broadcasting: shapes are normalised to rank 4 by prepending
//!   1s if needed (matches Conv's CHW handling), then dim-by-dim broadcast
//!   compatibility is checked. Per-channel BN-scale `[C, 1, 1]` × `[C, H, W]`
//!   data input lands here cleanly: prepend → `[1, C, 1, 1]` × `[1, C, H, W]`,
//!   broadcast → `[1, C, H, W]`.
//! - MIL `mb.add` / `mb.mul` / `mb.maximum` all natively broadcast in the
//!   ranks we emit, so no explicit `mb.broadcast_to` op is needed.

use std::collections::HashMap;

use anyhow::{Result, bail};

use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::logic::CompEq;
use tract_core::ops::math::{Add, Div, Max, Min, Mul, Sub};

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

/// Which element-wise op this is. Determines the MIL op type string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Mul,
    Max,
    Min,
    Sub,
    Div,
    /// Comparison `a == b` — output is bool, not float. Used in LLM mask
    /// construction (e.g. SmolLM2's attention mask is built via
    /// `Eq(positions, padding_id)` then routed through Iff).
    Eq,
}

impl BinOpKind {
    fn mil_op_type(self) -> &'static str {
        match self {
            BinOpKind::Add => "add",
            BinOpKind::Mul => "mul",
            BinOpKind::Max => "maximum",
            BinOpKind::Min => "minimum",
            BinOpKind::Sub => "sub",
            // MIL has both `real_div` (float division) and `floor_div` (integer
            // division). For F16 BinOp we want real_div.
            BinOpKind::Div => "real_div",
            BinOpKind::Eq => "equal",
        }
    }

    /// Output dtype. F16 for arithmetic; Bool for comparison.
    fn output_dtype(self) -> DataType {
        match self {
            BinOpKind::Eq => DataType::Bool,
            _ => DataType::Float16,
        }
    }
}

/// One input to a binop: either a runtime data value (wired by name) or a
/// compile-time constant (absorbed into the MLPackage weight blob).
pub enum BinOpInput {
    /// Value produced by another op (or the MLPackage external input).
    Data {
        /// MLPackage shape (rank 4 after normalisation).
        shape: Vec<i64>,
    },
    /// Compile-time constant absorbed into the weight blob.
    Const {
        /// MLPackage shape (rank 4 after normalisation).
        shape: Vec<i64>,
        /// FP16 tensor bytes (will be written to the blob at emit time).
        tensor: Tensor,
    },
}

#[allow(clippy::large_enum_variant)]
pub enum BinOpAnalysis {
    Translatable(BinOpPlan),
    Skip(String),
}

/// All the data needed to emit a MIL `add` / `mul` / `maximum`.
pub struct BinOpPlan {
    pub kind: BinOpKind,
    pub a: BinOpInput,
    pub b: BinOpInput,
    /// MLPackage output shape (rank 4 broadcast result).
    pub output_shape: Vec<i64>,
    /// tract-side output fact (may differ in rank from `output_shape` when
    /// CHW data format strips the batch dim; CoremlOp reshapes between).
    pub output_fact: TypedFact,
}

/// Identify the inner mini-op kind of a `TypedBinOp` node, if it's one of
/// the element-wise ops we translate.
pub fn binop_kind(node: &TypedNode) -> Option<BinOpKind> {
    let bin = node.op_as::<TypedBinOp>()?;
    let inner = bin.0.as_ref();
    if inner.downcast_ref::<Add>().is_some() {
        Some(BinOpKind::Add)
    } else if inner.downcast_ref::<Mul>().is_some() {
        Some(BinOpKind::Mul)
    } else if inner.downcast_ref::<Max>().is_some() {
        Some(BinOpKind::Max)
    } else if inner.downcast_ref::<Min>().is_some() {
        Some(BinOpKind::Min)
    } else if inner.downcast_ref::<Sub>().is_some() {
        Some(BinOpKind::Sub)
    } else if inner.downcast_ref::<Div>().is_some() {
        Some(BinOpKind::Div)
    } else if inner.downcast_ref::<CompEq>().is_some() {
        Some(BinOpKind::Eq)
    } else {
        None
    }
}

pub fn analyse_binop(model: &TypedModel, node: &TypedNode) -> Result<BinOpAnalysis> {
    let Some(kind) = binop_kind(node) else {
        return Ok(BinOpAnalysis::Skip("not a TypedBinOp(Add|Mul|Max|Min|Sub)".into()));
    };
    if node.inputs.len() != 2 {
        return Ok(BinOpAnalysis::Skip(format!(
            "{kind:?} node has {} inputs (need 2)",
            node.inputs.len()
        )));
    }

    let a = match analyse_input(model, node.inputs[0], "a")? {
        Ok(v) => v,
        Err(reason) => {
            return Ok(BinOpAnalysis::Skip(format!("{kind:?} input {reason}")));
        }
    };
    let b = match analyse_input(model, node.inputs[1], "b")? {
        Ok(v) => v,
        Err(reason) => {
            return Ok(BinOpAnalysis::Skip(format!("{kind:?} input {reason}")));
        }
    };

    let a_shape = input_shape(&a);
    let b_shape = input_shape(&b);
    let output_shape = match broadcast(a_shape, b_shape) {
        Some(r) => r,
        None => {
            return Ok(BinOpAnalysis::Skip(format!(
                "{kind:?} inputs not broadcast-compatible: {a_shape:?} vs {b_shape:?}"
            )));
        }
    };

    let output_fact = node.outputs[0].fact.clone();
    Ok(BinOpAnalysis::Translatable(BinOpPlan { kind, a, b, output_shape, output_fact }))
}

/// Inspect one input outlet: either a data input (with concrete F16 shape)
/// or a constant (resolved via [`const_tensor`] which walks `Cast(Const)`).
/// Returns Ok(Ok(input)) on success, Ok(Err(reason)) on Skip-able failure.
fn analyse_input(
    model: &TypedModel,
    outlet: OutletId,
    label: &str,
) -> Result<std::result::Result<BinOpInput, String>> {
    if let Some(t) = const_tensor(model, outlet)? {
        if t.datum_type() != DatumType::F16 {
            return Ok(Err(format!("{label} const dtype {:?} (need F16)", t.datum_type())));
        }
        let raw_shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
        let shape = super::rank::pad_to_rank_4(&raw_shape);
        return Ok(Ok(BinOpInput::Const { shape, tensor: t.into_owned() }));
    }
    let fact = model.outlet_fact(outlet)?;
    if fact.datum_type != DatumType::F16 {
        return Ok(Err(format!("{label} data dtype {:?} (need F16)", fact.datum_type)));
    }
    let raw_shape = match shape_to_concrete_i64(&fact.shape) {
        Some(s) => s,
        None => {
            return Ok(Err(format!("{label} data symbolic shape: {:?}", fact.shape)));
        }
    };
    if raw_shape.len() > 4 {
        return Ok(Err(format!("{label} data rank {} (only ≤4 supported)", raw_shape.len())));
    }
    let shape = super::rank::pad_to_rank_4(&raw_shape);
    Ok(Ok(BinOpInput::Data { shape }))
}

fn input_shape(i: &BinOpInput) -> &[i64] {
    match i {
        BinOpInput::Data { shape } => shape,
        BinOpInput::Const { shape, .. } => shape,
    }
}

/// NumPy-style broadcast: both inputs must already be the same rank (caller
/// pads via [`super::rank::pad_to_rank_4`]). Returns `None` if any dim
/// mismatches and neither side is 1.
fn broadcast(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut r = vec![0i64; a.len()];
    for i in 0..a.len() {
        r[i] = if a[i] == b[i] {
            a[i]
        } else if a[i] == 1 {
            b[i]
        } else if b[i] == 1 {
            a[i]
        } else {
            return None;
        };
    }
    Some(r)
}

/// Emit the MIL ops for a single tract `TypedBinOp(Add|Mul|Max)` translation.
///
/// `in_a_data_name` and `in_b_data_name` carry names ONLY when the
/// corresponding input is `BinOpInput::Data` (the caller looks them up via
/// `name_for`). For `BinOpInput::Const` inputs, the name is `None` and this
/// function emits the const as a fresh `op_const_blob` MIL op naming it
/// `{output_name}_const_a` / `_const_b` and writes the tensor bytes to the
/// shared `BlobBuilder`.
pub fn emit_binop_mil(
    plan: &BinOpPlan,
    blob: &mut BlobBuilder,
    in_a_data_name: Option<&str>,
    in_b_data_name: Option<&str>,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let mut mil_ops: Vec<mil::Operation> = Vec::new();

    let a_name = resolve_input_name(&plan.a, in_a_data_name, blob, output_name, "a", &mut mil_ops)?;
    let b_name = resolve_input_name(&plan.b, in_b_data_name, blob, output_name, "b", &mut mil_ops)?;

    // Defensive input-rank normalization. Same pattern as Conv / Concat /
    // Slice / Activation / RmsNorm: analyse_binop pads each operand to
    // rank-4 via `pad_to_rank_4`, but the upstream MIL value may be at a
    // different rank (e.g. a Gather output that emits at tract's natural
    // rank-3). MIL `add` / `mul` / etc. have NumPy-style broadcast inference
    // that would produce a rank-3 output if both inputs are rank-3, but we
    // declared rank-4 — the resulting type mismatch aborts MLPackage
    // compile with "Output '0' has unexpected type 'ios17.add'. Expected
    // tensor<fp16, [...rank-3]>; got tensor<fp16, [1, ...rank-3]>". Surfaced
    // on SmolLM2 once Gather + Cos + Sin started translating the embedding
    // and RoPE-table outputs at tract's natural rank-3.
    let a_rank4_name = format!("{output_name}_a_rank{}", plan_input_shape(&plan.a).len());
    let a_shape_n = format!("{output_name}_a_shape");
    let a_shape_op = op_const_immediate(
        &a_shape_n,
        tensor_type(DataType::Int32, &[plan_input_shape(&plan.a).len() as i64]),
        tv_ints(plan_input_shape(&plan.a).iter().map(|&v| v as i32).collect()),
    );
    let a_rank4_ty = tensor_type(DataType::Float16, plan_input_shape(&plan.a));
    mil_ops.push(a_shape_op);
    mil_ops.push(mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&a_name)),
            ("shape".into(), arg_name(&a_shape_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: a_rank4_name.clone(), r#type: Some(a_rank4_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    let b_rank4_name = format!("{output_name}_b_rank{}", plan_input_shape(&plan.b).len());
    let b_shape_n = format!("{output_name}_b_shape");
    let b_shape_op = op_const_immediate(
        &b_shape_n,
        tensor_type(DataType::Int32, &[plan_input_shape(&plan.b).len() as i64]),
        tv_ints(plan_input_shape(&plan.b).iter().map(|&v| v as i32).collect()),
    );
    let b_rank4_ty = tensor_type(DataType::Float16, plan_input_shape(&plan.b));
    mil_ops.push(b_shape_op);
    mil_ops.push(mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&b_name)),
            ("shape".into(), arg_name(&b_shape_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: b_rank4_name.clone(), r#type: Some(b_rank4_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // Output dtype follows the BinOp kind: F16 for arithmetic, Bool for
    // comparisons (Eq, etc.). MIL infers this anyway from the op type, but
    // we declare it explicitly so downstream consumers see the right type.
    let out_ty = tensor_type(plan.kind.output_dtype(), &plan.output_shape);
    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(&a_rank4_name));
    inputs.insert("y".into(), arg_name(&b_rank4_name));

    let op = mil::Operation {
        r#type: plan.kind.mil_op_type().to_string(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    mil_ops.push(op);
    Ok(mil_ops)
}

/// Get the rank-4-padded shape of a `BinOpInput` (works for both Data and Const).
fn plan_input_shape(i: &BinOpInput) -> &[i64] {
    match i {
        BinOpInput::Data { shape } | BinOpInput::Const { shape, .. } => shape,
    }
}

fn resolve_input_name(
    src: &BinOpInput,
    data_name: Option<&str>,
    blob: &mut BlobBuilder,
    output_name: &str,
    label: &str,
    mil_ops: &mut Vec<mil::Operation>,
) -> Result<String> {
    match (src, data_name) {
        (BinOpInput::Data { .. }, Some(n)) => Ok(n.to_string()),
        (BinOpInput::Data { .. }, None) => {
            bail!("BinOp::emit: input {label} is Data but no name passed by caller")
        }
        (BinOpInput::Const { shape, tensor }, None) => {
            let const_name = format!("{output_name}_const_{label}");
            let const_ty = tensor_type(DataType::Float16, shape);
            let offset = blob.add(BlobDataType::Float16, tensor.as_bytes());
            let const_op =
                op_const_blob(&const_name, const_ty, mlpackage::WEIGHT_BLOB_PATH, offset);
            mil_ops.push(const_op);
            Ok(const_name)
        }
        (BinOpInput::Const { .. }, Some(_)) => bail!(
            "BinOp::emit: input {label} is Const but caller passed a name (probably a fusion bug)"
        ),
    }
}

//! tract `Iff` (if-then-else) → MIL `select` translator.
//!
//! `Iff(cond, t, f)` returns elementwise `cond ? t : f`. tract requires
//! `cond` to be Bool and `t`/`f` to share the same dtype. Surfaces in LLM
//! mask construction (e.g. SmolLM2's attention-mask routing). Pairs with
//! the `BinOpKind::Eq` comparison to make a complete mask-building chain
//! translatable.
//!
//! MIL has `mb.select(cond, a, b)` with the same semantics; rank-aligned
//! NumPy-style broadcasting across the three inputs.

use std::collections::HashMap;

use anyhow::{Result, bail};

use tract_core::internal::*;
use tract_core::ops::logic::Iff;

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum IffAnalysis {
    Translatable(IffPlan),
    Skip(String),
}

/// One Iff slot: either a runtime Data input (wired by name) or a Const
/// (absorbed inline). For `cond` the const tensor is bool; for `t`/`f`
/// it's F16. Same enum pattern as BinOp / Concat / Gather.
pub enum IffInput {
    Data { shape: Vec<i64>, datum_type: DatumType },
    Const { shape: Vec<i64>, tensor: Tensor },
}

pub struct IffPlan {
    pub cond: IffInput,
    pub t: IffInput,
    pub f: IffInput,
    /// Broadcast output shape (max-rank elementwise).
    pub output_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_iff(model: &TypedModel, node: &TypedNode) -> Result<IffAnalysis> {
    if node.op_as::<Iff>().is_none() {
        return Ok(IffAnalysis::Skip("not an Iff".into()));
    }
    if node.inputs.len() != 3 {
        return Ok(IffAnalysis::Skip(format!(
            "Iff has {} inputs (need 3: cond + t + f)",
            node.inputs.len()
        )));
    }

    let cond = match analyse_input(model, node.inputs[0], "cond", &[DatumType::Bool])? {
        Ok(v) => v,
        Err(reason) => return Ok(IffAnalysis::Skip(format!("Iff cond {reason}"))),
    };
    let t = match analyse_input(model, node.inputs[1], "t", &[DatumType::F16])? {
        Ok(v) => v,
        Err(reason) => return Ok(IffAnalysis::Skip(format!("Iff t {reason}"))),
    };
    let f = match analyse_input(model, node.inputs[2], "f", &[DatumType::F16])? {
        Ok(v) => v,
        Err(reason) => return Ok(IffAnalysis::Skip(format!("Iff f {reason}"))),
    };

    // Compute the broadcast output shape across all three inputs (max rank,
    // dim-by-dim broadcast on aligned suffixes).
    let cond_shape = match &cond {
        IffInput::Data { shape, .. } | IffInput::Const { shape, .. } => shape.clone(),
    };
    let t_shape = match &t {
        IffInput::Data { shape, .. } | IffInput::Const { shape, .. } => shape.clone(),
    };
    let f_shape = match &f {
        IffInput::Data { shape, .. } | IffInput::Const { shape, .. } => shape.clone(),
    };
    let output_shape = match broadcast3(&cond_shape, &t_shape, &f_shape) {
        Some(s) => s,
        None => {
            return Ok(IffAnalysis::Skip(format!(
                "Iff inputs not broadcast-compatible: cond={cond_shape:?} t={t_shape:?} f={f_shape:?}"
            )));
        }
    };

    Ok(IffAnalysis::Translatable(IffPlan {
        cond,
        t,
        f,
        output_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

fn analyse_input(
    model: &TypedModel,
    outlet: OutletId,
    label: &str,
    accepted_dtypes: &[DatumType],
) -> Result<std::result::Result<IffInput, String>> {
    if let Some(t) = const_tensor(model, outlet)? {
        if !accepted_dtypes.contains(&t.datum_type()) {
            return Ok(Err(format!(
                "{label} const dtype {:?} (accepted: {:?})",
                t.datum_type(),
                accepted_dtypes
            )));
        }
        let shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
        return Ok(Ok(IffInput::Const { shape, tensor: t.into_owned() }));
    }
    let fact = model.outlet_fact(outlet)?;
    if !accepted_dtypes.contains(&fact.datum_type) {
        return Ok(Err(format!(
            "{label} data dtype {:?} (accepted: {:?})",
            fact.datum_type, accepted_dtypes
        )));
    }
    let shape = match shape_to_concrete_i64(&fact.shape) {
        Some(s) => s,
        None => return Ok(Err(format!("{label} symbolic shape: {:?}", fact.shape))),
    };
    Ok(Ok(IffInput::Data { shape, datum_type: fact.datum_type }))
}

/// Three-way broadcast for cond/t/f shapes. Pads each to the max rank by
/// prepending 1s, then per-dim picks the non-1 dim (or fails if any pair
/// disagrees on a non-1 dim).
fn broadcast3(a: &[i64], b: &[i64], c: &[i64]) -> Option<Vec<i64>> {
    let r = a.len().max(b.len()).max(c.len());
    let pad = |s: &[i64]| -> Vec<i64> {
        let mut v = vec![1i64; r - s.len()];
        v.extend_from_slice(s);
        v
    };
    let a = pad(a);
    let b = pad(b);
    let c = pad(c);
    let mut out = vec![0i64; r];
    for i in 0..r {
        let dims = [a[i], b[i], c[i]];
        let max = *dims.iter().max().unwrap();
        for &d in &dims {
            if d != 1 && d != max {
                return None;
            }
        }
        out[i] = max;
    }
    Some(out)
}

/// Emit MIL ops for an Iff. Per-slot data names: `Some(name)` if the slot
/// is `IffInput::Data`, `None` if it's `IffInput::Const` (absorbed inline).
pub fn emit_iff_mil(
    plan: &IffPlan,
    blob: &mut BlobBuilder,
    cond_data_name: Option<&str>,
    t_data_name: Option<&str>,
    f_data_name: Option<&str>,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let mut prelude: Vec<mil::Operation> = Vec::new();

    let cond_name =
        resolve_input(&plan.cond, cond_data_name, blob, output_name, "cond", &mut prelude)?;
    let t_name = resolve_input(&plan.t, t_data_name, blob, output_name, "t", &mut prelude)?;
    let f_name = resolve_input(&plan.f, f_data_name, blob, output_name, "f", &mut prelude)?;

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("cond".into(), arg_name(&cond_name));
    inputs.insert("a".into(), arg_name(&t_name));
    inputs.insert("b".into(), arg_name(&f_name));

    let op = mil::Operation {
        r#type: "select".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(op);
    Ok(prelude)
}

fn resolve_input(
    src: &IffInput,
    data_name: Option<&str>,
    blob: &mut BlobBuilder,
    output_name: &str,
    label: &str,
    mil_ops: &mut Vec<mil::Operation>,
) -> Result<String> {
    match (src, data_name) {
        (IffInput::Data { .. }, Some(n)) => Ok(n.to_string()),
        (IffInput::Data { .. }, None) => {
            bail!("Iff::emit: input {label} is Data but caller passed no name (fusion bug)")
        }
        (IffInput::Const { shape, tensor }, None) => {
            let const_name = format!("{output_name}_{label}_const");
            // Pick the right BlobDataType / MIL DataType from the const dtype.
            let (blob_dt, mil_dt) = match tensor.datum_type() {
                DatumType::F16 => (BlobDataType::Float16, DataType::Float16),
                DatumType::Bool => {
                    // Bool consts use op_const_immediate with RepeatedBytes;
                    // skip blob path entirely. Common for an all-True / all-
                    // False mask absorbed at compile time.
                    let bytes: Vec<u8> = unsafe {
                        tensor.as_slice_unchecked::<bool>().iter().map(|&b| b as u8).collect()
                    };
                    let const_op = op_const_immediate(
                        &const_name,
                        tensor_type(DataType::Bool, shape),
                        mil::TensorValue {
                            value: Some(mil::tensor_value::Value::Bytes(
                                mil::tensor_value::RepeatedBytes { values: bytes },
                            )),
                        },
                    );
                    mil_ops.push(const_op);
                    return Ok(const_name);
                }
                other => bail!("Iff::emit: const dtype {other:?} not yet supported for {label}"),
            };
            let const_ty = tensor_type(mil_dt, shape);
            let offset = blob.add(blob_dt, tensor.as_bytes());
            let const_op =
                op_const_blob(&const_name, const_ty, mlpackage::WEIGHT_BLOB_PATH, offset);
            mil_ops.push(const_op);
            Ok(const_name)
        }
        (IffInput::Const { .. }, Some(_)) => {
            bail!("Iff::emit: input {label} is Const but caller passed a name (fusion bug)")
        }
    }
}

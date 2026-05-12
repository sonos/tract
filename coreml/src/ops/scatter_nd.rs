//! tract `ScatterNd` → MIL `scatter_nd` translator.
//!
//! `ScatterNd(data, indices, updates)` writes `updates` into `data` at
//! locations specified by `indices`, returning a new tensor of `data`'s
//! shape. ONNX semantics: `indices` is rank-K with last dim Q (= how many
//! of `data`'s leading dims to index); `updates` is rank `(K-1) + (data.rank
//! - Q)`. tract supports an optional reduction mode (None / Add / Mul /
//! Min / Max); MIL `scatter_nd` accepts the same set.
//!
//! Surfaces in LLMs at the position-id construction step: typically
//! `scatter_nd(zeros, [[0]], range(seq_len))` materialises position IDs
//! `[0, 1, 2, ..., S-1]` in a single op. Without translation, the model
//! fragments at the entry — preventing the embedding lookup + attention
//! chain from fusing into one CoremlOp.
//!
//! Implementation notes:
//! - All three inputs may be Const (typical when position IDs are computed
//!   from compile-time-known shapes) or runtime Data; same enum pattern as
//!   `BinOp` / `Concat` / `Gather` / `Iff`.
//! - I64 indices are cast to I32 for MIL via the same boundary mechanism as
//!   Gather (CoremlOp eval-side cast for runtime Data; inline conversion
//!   for Const).

use std::collections::HashMap;

use anyhow::{Result, bail};

use tract_core::internal::*;
use tract_core::ops::array::{ScatterNd, ScatterReduction};

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints, tv_strings};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum ScatterNdAnalysis {
    Translatable(ScatterNdPlan),
    Skip(String),
}

pub enum ScatterNdInput {
    Data { shape: Vec<i64>, datum_type: DatumType },
    Const { shape: Vec<i64>, tensor: Tensor },
}

pub struct ScatterNdPlan {
    pub data: ScatterNdInput,
    pub indices: ScatterNdInput,
    pub updates: ScatterNdInput,
    pub reduction: ScatterReduction,
    /// Output shape (== data shape).
    pub output_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_scatter_nd(model: &TypedModel, node: &TypedNode) -> Result<ScatterNdAnalysis> {
    let Some(sn) = node.op_as::<ScatterNd>() else {
        return Ok(ScatterNdAnalysis::Skip("not a ScatterNd".into()));
    };
    if node.inputs.len() != 3 {
        return Ok(ScatterNdAnalysis::Skip(format!(
            "ScatterNd has {} inputs (need 3: data + indices + updates)",
            node.inputs.len()
        )));
    }

    let data = match analyse_input(model, node.inputs[0], "data", &[DatumType::F16])? {
        Ok(v) => v,
        Err(reason) => return Ok(ScatterNdAnalysis::Skip(format!("ScatterNd data {reason}"))),
    };
    let indices =
        match analyse_input(model, node.inputs[1], "indices", &[DatumType::I64, DatumType::I32])? {
            Ok(v) => v,
            Err(reason) => {
                return Ok(ScatterNdAnalysis::Skip(format!("ScatterNd indices {reason}")));
            }
        };
    let updates = match analyse_input(model, node.inputs[2], "updates", &[DatumType::F16])? {
        Ok(v) => v,
        Err(reason) => return Ok(ScatterNdAnalysis::Skip(format!("ScatterNd updates {reason}"))),
    };

    let data_shape = match &data {
        ScatterNdInput::Data { shape, .. } | ScatterNdInput::Const { shape, .. } => shape.clone(),
    };

    Ok(ScatterNdAnalysis::Translatable(ScatterNdPlan {
        data,
        indices,
        updates,
        reduction: sn.reduction,
        output_shape: data_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

fn analyse_input(
    model: &TypedModel,
    outlet: OutletId,
    label: &str,
    accepted_dtypes: &[DatumType],
) -> Result<std::result::Result<ScatterNdInput, String>> {
    if let Some(t) = const_tensor(model, outlet)? {
        if !accepted_dtypes.contains(&t.datum_type()) {
            return Ok(Err(format!(
                "{label} const dtype {:?} (accepted: {:?})",
                t.datum_type(),
                accepted_dtypes
            )));
        }
        let shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
        return Ok(Ok(ScatterNdInput::Const { shape, tensor: t.into_owned() }));
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
    Ok(Ok(ScatterNdInput::Data { shape, datum_type: fact.datum_type }))
}

/// Map tract's reduction mode to MIL's `mode` string.
fn mil_mode(r: ScatterReduction) -> &'static str {
    match r {
        ScatterReduction::None => "update",
        ScatterReduction::Add => "add",
        ScatterReduction::Mul => "mul",
        ScatterReduction::Min => "min",
        ScatterReduction::Max => "max",
    }
}

pub fn emit_scatter_nd_mil(
    plan: &ScatterNdPlan,
    blob: &mut BlobBuilder,
    data_data_name: Option<&str>,
    indices_data_name: Option<&str>,
    updates_data_name: Option<&str>,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let mut prelude: Vec<mil::Operation> = Vec::new();

    // ---- Resolve `data` (F16). Same pattern as Gather.
    let data_name = match (&plan.data, data_data_name) {
        (ScatterNdInput::Data { .. }, Some(n)) => n.to_string(),
        (ScatterNdInput::Data { .. }, None) => {
            bail!("ScatterNd::emit: data is Data but caller passed no name (fusion bug)")
        }
        (ScatterNdInput::Const { shape, tensor }, None) => {
            let const_name = format!("{output_name}_data_const");
            let const_ty = tensor_type(DataType::Float16, shape);
            let offset = blob.add(BlobDataType::Float16, tensor.as_bytes());
            let const_op =
                op_const_blob(&const_name, const_ty, mlpackage::WEIGHT_BLOB_PATH, offset);
            prelude.push(const_op);
            const_name
        }
        (ScatterNdInput::Const { .. }, Some(_)) => {
            bail!("ScatterNd::emit: data is Const but caller passed a name (fusion bug)")
        }
    };

    // ---- Resolve `indices`. MIL expects I32; cast I64 → I32 if needed.
    let indices_name = match (&plan.indices, indices_data_name) {
        (ScatterNdInput::Data { datum_type, shape }, Some(n)) => {
            if *datum_type == DatumType::I32 {
                n.to_string()
            } else {
                // Cast I64 → I32. MIL `cast` is metadata-only when values fit.
                let cast_n = format!("{output_name}_indices_i32");
                let cast_dt_n = format!("{output_name}_indices_dtype");
                let cast_dt_op = op_const_immediate(
                    &cast_dt_n,
                    tensor_type_scalar(DataType::String),
                    tv_strings(vec!["int32".into()]),
                );
                let mut cast_inputs: HashMap<String, mil::Argument> = HashMap::new();
                cast_inputs.insert("x".into(), arg_name(n));
                cast_inputs.insert("dtype".into(), arg_name(&cast_dt_n));
                let cast_op = mil::Operation {
                    r#type: "cast".into(),
                    inputs: cast_inputs,
                    outputs: vec![mil::NamedValueType {
                        name: cast_n.clone(),
                        r#type: Some(tensor_type(DataType::Int32, shape)),
                    }],
                    blocks: vec![],
                    attributes: HashMap::new(),
                };
                prelude.push(cast_dt_op);
                prelude.push(cast_op);
                cast_n
            }
        }
        (ScatterNdInput::Data { .. }, None) => {
            bail!("ScatterNd::emit: indices is Data but caller passed no name (fusion bug)")
        }
        (ScatterNdInput::Const { shape, tensor }, None) => {
            let const_n = format!("{output_name}_indices_const");
            let i32_values: Vec<i32> = match tensor.datum_type() {
                DatumType::I64 => unsafe {
                    tensor.as_slice_unchecked::<i64>().iter().map(|&v| v as i32).collect()
                },
                DatumType::I32 => unsafe { tensor.as_slice_unchecked::<i32>().to_vec() },
                other => bail!("ScatterNd const indices dtype {other:?} not supported"),
            };
            let const_op = op_const_immediate(
                &const_n,
                tensor_type(DataType::Int32, shape),
                tv_ints(i32_values),
            );
            prelude.push(const_op);
            const_n
        }
        (ScatterNdInput::Const { .. }, Some(_)) => {
            bail!("ScatterNd::emit: indices is Const but caller passed a name (fusion bug)")
        }
    };

    // ---- Resolve `updates` (F16). Same pattern as data.
    let updates_name = match (&plan.updates, updates_data_name) {
        (ScatterNdInput::Data { .. }, Some(n)) => n.to_string(),
        (ScatterNdInput::Data { .. }, None) => {
            bail!("ScatterNd::emit: updates is Data but caller passed no name (fusion bug)")
        }
        (ScatterNdInput::Const { shape, tensor }, None) => {
            let const_name = format!("{output_name}_updates_const");
            let const_ty = tensor_type(DataType::Float16, shape);
            let offset = blob.add(BlobDataType::Float16, tensor.as_bytes());
            let const_op =
                op_const_blob(&const_name, const_ty, mlpackage::WEIGHT_BLOB_PATH, offset);
            prelude.push(const_op);
            const_name
        }
        (ScatterNdInput::Const { .. }, Some(_)) => {
            bail!("ScatterNd::emit: updates is Const but caller passed a name (fusion bug)")
        }
    };

    // ---- Mode const.
    let mode_n = format!("{output_name}_mode");
    let mode_op = op_const_immediate(
        &mode_n,
        tensor_type_scalar(DataType::String),
        tv_strings(vec![mil_mode(plan.reduction).into()]),
    );
    prelude.push(mode_op);

    // ---- Validate const (required by MIL).
    let validate_n = format!("{output_name}_validate_indices");
    let validate_op = op_const_immediate(
        &validate_n,
        tensor_type_scalar(DataType::Bool),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: vec![false],
            })),
        },
    );
    prelude.push(validate_op);

    // ---- The MIL scatter_nd op. Output shape == data shape (F16).
    let mut s_inputs: HashMap<String, mil::Argument> = HashMap::new();
    s_inputs.insert("data".into(), arg_name(&data_name));
    s_inputs.insert("indices".into(), arg_name(&indices_name));
    s_inputs.insert("updates".into(), arg_name(&updates_name));
    s_inputs.insert("mode".into(), arg_name(&mode_n));
    s_inputs.insert("validate_indices".into(), arg_name(&validate_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let s_op = mil::Operation {
        r#type: "scatter_nd".into(),
        inputs: s_inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(s_op);

    Ok(prelude)
}

//! tract `Gather` → MIL `gather` translator.
//!
//! Two distinct use patterns observed in the canary corpus:
//!
//! - **Embedding lookup** (SmolLM2-class LLMs, position 0 of the model):
//!     * `data` is a const F16 embedding matrix `[vocab_size, hidden]`
//!     * `indices` is a runtime I64 tensor of token IDs `[batch, seq_len]`
//!     * `axis = 0`, output `[batch, seq_len, hidden]`
//!   Without a translator, every LLM strands the embedding lookup at the
//!   model entry, fragmenting the entire downstream attention chain into a
//!   separate CoremlOp.
//!
//! - **CLS-token pluck** (ViT, end of the model):
//!     * `data` is a runtime F16 encoder output `[1, num_patches+1, hidden]`
//!     * `indices` is a scalar I64 const `0`
//!     * `axis = 1`, output `[1, hidden]`
//!   ViT had this as the only stranded compute op after Phase 5's bug
//!   fixes; with this translator it folds into the encoder's CoremlOp.
//!
//! Either input can be Const or Data — same enum pattern as `BinOp` /
//! `Concat`. MIL's `gather` accepts I32 indices; we cast I64 → I32 inline
//! when the indices come from a runtime input (no-op if upstream already
//! cast — caller-side concern).

use std::collections::HashMap;

use anyhow::{Result, bail};

use tract_core::internal::*;
use tract_core::ops::array::Gather;

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum GatherAnalysis {
    Translatable(GatherPlan),
    Skip(String),
}

/// Per-input slot classification — Data (wired by name from upstream) or
/// Const (resolved at analyse time, absorbed into the MLPackage at emit).
pub enum GatherInput {
    Data { shape: Vec<i64>, datum_type: DatumType },
    Const { shape: Vec<i64>, tensor: Tensor },
}

pub struct GatherPlan {
    pub data: GatherInput,
    pub indices: GatherInput,
    /// Axis being gathered along (in `data`'s rank coords).
    pub axis: i64,
    /// Output shape (`data.shape[..axis] ++ indices.shape ++ data.shape[axis+1..]`).
    pub output_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_gather(model: &TypedModel, node: &TypedNode) -> Result<GatherAnalysis> {
    let Some(g) = node.op_as::<Gather>() else {
        return Ok(GatherAnalysis::Skip("not a Gather".into()));
    };
    if node.inputs.len() != 2 {
        return Ok(GatherAnalysis::Skip(format!(
            "Gather has {} inputs (need 2: data + indices)",
            node.inputs.len()
        )));
    }

    let data_in = analyse_input(model, node.inputs[0], "data", &[DatumType::F16])?;
    let data = match data_in {
        Ok(v) => v,
        Err(reason) => return Ok(GatherAnalysis::Skip(format!("Gather data {reason}"))),
    };
    // Indices can be I64 (typical ONNX) or I32; we'll cast to I32 at emit.
    let indices_in =
        analyse_input(model, node.inputs[1], "indices", &[DatumType::I64, DatumType::I32])?;
    let indices = match indices_in {
        Ok(v) => v,
        Err(reason) => return Ok(GatherAnalysis::Skip(format!("Gather indices {reason}"))),
    };

    let data_shape = match &data {
        GatherInput::Data { shape, .. } | GatherInput::Const { shape, .. } => shape,
    };
    let indices_shape = match &indices {
        GatherInput::Data { shape, .. } | GatherInput::Const { shape, .. } => shape,
    };

    if g.axis >= data_shape.len() {
        return Ok(GatherAnalysis::Skip(format!(
            "Gather axis {} >= data rank {}",
            g.axis,
            data_shape.len()
        )));
    }

    // Output shape = data.shape[..axis] ++ indices.shape ++ data.shape[axis+1..]
    let mut output_shape: Vec<i64> = Vec::new();
    output_shape.extend_from_slice(&data_shape[..g.axis]);
    output_shape.extend_from_slice(indices_shape);
    output_shape.extend_from_slice(&data_shape[g.axis + 1..]);

    Ok(GatherAnalysis::Translatable(GatherPlan {
        data,
        indices,
        axis: g.axis as i64,
        output_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

fn analyse_input(
    model: &TypedModel,
    outlet: OutletId,
    label: &str,
    accepted_dtypes: &[DatumType],
) -> Result<std::result::Result<GatherInput, String>> {
    if let Some(t) = const_tensor(model, outlet)? {
        if !accepted_dtypes.contains(&t.datum_type()) {
            return Ok(Err(format!(
                "{label} const dtype {:?} (accepted: {:?})",
                t.datum_type(),
                accepted_dtypes
            )));
        }
        let shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
        return Ok(Ok(GatherInput::Const { shape, tensor: t.into_owned() }));
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
    Ok(Ok(GatherInput::Data { shape, datum_type: fact.datum_type }))
}

/// Emit MIL ops for a Gather. `data_data_name` and `indices_data_name`:
/// `Some(name)` for a Data input the caller wired by name (must match
/// `GatherInput::Data` at that slot); `None` for a Const input that emit
/// handles inline (must match `GatherInput::Const`).
pub fn emit_gather_mil(
    plan: &GatherPlan,
    blob: &mut BlobBuilder,
    data_data_name: Option<&str>,
    indices_data_name: Option<&str>,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let mut prelude: Vec<mil::Operation> = Vec::new();

    // Resolve `data` input — write the const blob if it's Const, otherwise
    // use the caller-supplied name.
    let data_name: String = match (&plan.data, data_data_name) {
        (GatherInput::Data { .. }, Some(n)) => n.to_string(),
        (GatherInput::Data { .. }, None) => {
            bail!("Gather::emit: data input is Data but caller passed no name (fusion bug)")
        }
        (GatherInput::Const { shape, tensor }, None) => {
            let const_name = format!("{output_name}_data_const");
            // Embedding tables are F16 in our pipeline (post-`f32_to_f16`).
            let dt = match tensor.datum_type() {
                DatumType::F16 => BlobDataType::Float16,
                other => bail!("Gather const data dtype {other:?} not yet supported"),
            };
            let const_ty = tensor_type(DataType::Float16, shape);
            let offset = blob.add(dt, tensor.as_bytes());
            let const_op =
                op_const_blob(&const_name, const_ty, mlpackage::WEIGHT_BLOB_PATH, offset);
            prelude.push(const_op);
            const_name
        }
        (GatherInput::Const { .. }, Some(_)) => {
            bail!("Gather::emit: data is Const but caller passed a name (fusion bug)")
        }
    };

    // Resolve `indices` input. MIL `gather` accepts int32 indices; we cast
    // I64 → I32 if needed for the runtime path. Const indices are emitted
    // as int32 immediates directly.
    let indices_name: String = match (&plan.indices, indices_data_name) {
        (GatherInput::Data { datum_type, .. }, Some(n)) => {
            if *datum_type == DatumType::I32 {
                n.to_string()
            } else {
                // Insert a cast to I32. This is a metadata-only typecast in
                // MIL when the underlying values fit in I32 (ONNX indices
                // are non-negative token IDs well under 2^31).
                let cast_n = format!("{output_name}_indices_i32");
                let cast_dt_n = format!("{output_name}_indices_dtype");
                let cast_dt_op = op_const_immediate(
                    &cast_dt_n,
                    tensor_type_scalar(DataType::String),
                    mil::TensorValue {
                        value: Some(mil::tensor_value::Value::Strings(
                            mil::tensor_value::RepeatedStrings { values: vec!["int32".into()] },
                        )),
                    },
                );
                let mut cast_inputs: HashMap<String, mil::Argument> = HashMap::new();
                cast_inputs.insert("x".into(), arg_name(n));
                cast_inputs.insert("dtype".into(), arg_name(&cast_dt_n));
                let indices_shape = match &plan.indices {
                    GatherInput::Data { shape, .. } | GatherInput::Const { shape, .. } => shape,
                };
                let cast_op = mil::Operation {
                    r#type: "cast".into(),
                    inputs: cast_inputs,
                    outputs: vec![mil::NamedValueType {
                        name: cast_n.clone(),
                        r#type: Some(tensor_type(DataType::Int32, indices_shape)),
                    }],
                    blocks: vec![],
                    attributes: HashMap::new(),
                };
                prelude.push(cast_dt_op);
                prelude.push(cast_op);
                cast_n
            }
        }
        (GatherInput::Data { .. }, None) => {
            bail!("Gather::emit: indices is Data but caller passed no name (fusion bug)")
        }
        (GatherInput::Const { shape, tensor }, None) => {
            let const_n = format!("{output_name}_indices_const");
            let i32_values: Vec<i32> = match tensor.datum_type() {
                DatumType::I64 => unsafe {
                    tensor.as_slice_unchecked::<i64>().iter().map(|&v| v as i32).collect()
                },
                DatumType::I32 => unsafe { tensor.as_slice_unchecked::<i32>().to_vec() },
                other => bail!("Gather const indices dtype {other:?} not supported"),
            };
            let const_op = op_const_immediate(
                &const_n,
                tensor_type(DataType::Int32, shape),
                tv_ints(i32_values),
            );
            prelude.push(const_op);
            const_n
        }
        (GatherInput::Const { .. }, Some(_)) => {
            bail!("Gather::emit: indices is Const but caller passed a name (fusion bug)")
        }
    };

    // Axis const.
    let axis_n = format!("{output_name}_axis");
    let axis_op = op_const_immediate(
        &axis_n,
        tensor_type_scalar(DataType::Int32),
        tv_ints(vec![plan.axis as i32]),
    );
    prelude.push(axis_op);

    // `validate_indices` is required by MIL `gather` (iOS17 opset). Pass
    // false — we trust the upstream model's indices to be in-range; runtime
    // validation would be a per-element check that defeats the point of
    // dispatching to ANE.
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

    // The MIL gather op.
    let mut g_inputs: HashMap<String, mil::Argument> = HashMap::new();
    g_inputs.insert("x".into(), arg_name(&data_name));
    g_inputs.insert("indices".into(), arg_name(&indices_name));
    g_inputs.insert("axis".into(), arg_name(&axis_n));
    g_inputs.insert("validate_indices".into(), arg_name(&validate_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let g_op = mil::Operation {
        r#type: "gather".into(),
        inputs: g_inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(g_op);

    Ok(prelude)
}

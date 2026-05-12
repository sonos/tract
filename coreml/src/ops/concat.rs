//! tract `TypedConcat` → MIL `concat` translator.
//!
//! Concat joins N inputs along a single axis. In SqueezeNet's fire modules,
//! the 1×1 and 3×3 expand convolutions are concatenated along the channel
//! axis to recombine the squeezed branches — without a Concat translator,
//! every fire-module boundary fragments the graph into separate MLPackages
//! (the same problem MobileNet had pre-EinSum at 47 CoremlOps).
//!
//! Phase 3 first cut: F16 only, all inputs same rank, axis interpreted in the
//! tract-side fact rank (rank-4 NCHW or rank-3 CHW); MLPackage-side rank is
//! always padded to 4 by prepending 1s, with the axis shifted accordingly.

use std::collections::HashMap;

use anyhow::{Result, bail};

use tract_core::internal::*;
use tract_core::ops::array::TypedConcat;

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, arg_names, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum ConcatAnalysis {
    Translatable(ConcatPlan),
    Skip(String),
}

/// One input slot to a Concat: either a runtime Data value (wired from
/// upstream by the caller) or a compile-time Const (resolved at analyse
/// time via [`const_tensor`] — handles `Cast(Const)` chains too — and
/// absorbed into the MLPackage weight blob at emit time). Same pattern as
/// `BinOpInput`. Surfaces in ViT's CLS-token Concat, where the CLS token is
/// a Const that gets concatenated with the patch embeddings; without
/// const-input handling here, the fusion driver fails with
/// "Concat node ... has unmapped input slot 0".
pub enum ConcatInput {
    Data { shape: Vec<i64> },
    Const { shape: Vec<i64>, tensor: Tensor },
}

pub struct ConcatPlan {
    /// One entry per input slot, in tract-input order. Each input's
    /// MLPackage-side shape is rank-4 (normalised by prepending 1s).
    pub inputs: Vec<ConcatInput>,
    /// Output MLPackage-side shape (rank 4).
    pub output_shape: Vec<i64>,
    /// Axis in MLPackage-side rank-4 coordinates (i.e. shifted by the prepend).
    pub axis: i64,
    /// tract-side output fact (may be rank 3 if CHW; CoremlOp reshapes).
    pub output_fact: TypedFact,
}

impl ConcatPlan {
    /// Convenience for callers that just want the per-slot rank-4 shapes.
    pub fn input_shapes(&self) -> Vec<&Vec<i64>> {
        self.inputs
            .iter()
            .map(|i| match i {
                ConcatInput::Data { shape } | ConcatInput::Const { shape, .. } => shape,
            })
            .collect()
    }
}

pub fn analyse_concat(model: &TypedModel, node: &TypedNode) -> Result<ConcatAnalysis> {
    let Some(concat) = node.op_as::<TypedConcat>() else {
        return Ok(ConcatAnalysis::Skip("not a TypedConcat".into()));
    };
    if node.inputs.is_empty() {
        return Ok(ConcatAnalysis::Skip("Concat has 0 inputs".into()));
    }

    // For each input slot: either resolve as a Const (walks Cast(Const)
    // chains) or treat as a runtime Data input with its outlet shape.
    // Both paths yield (raw_shape, optional Const tensor).
    let mut raw_shapes: Vec<Vec<i64>> = Vec::with_capacity(node.inputs.len());
    let mut const_tensors: Vec<Option<Tensor>> = Vec::with_capacity(node.inputs.len());
    for (slot, input) in node.inputs.iter().enumerate() {
        if let Some(t) = const_tensor(model, *input)? {
            if t.datum_type() != DatumType::F16 {
                return Ok(ConcatAnalysis::Skip(format!(
                    "Concat input {slot} const dtype {:?} (need F16)",
                    t.datum_type()
                )));
            }
            let raw_shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
            raw_shapes.push(raw_shape);
            const_tensors.push(Some(t.into_owned()));
        } else {
            let fact = model.outlet_fact(*input)?;
            if fact.datum_type != DatumType::F16 {
                return Ok(ConcatAnalysis::Skip(format!(
                    "Concat input {slot} data dtype {:?} (need F16)",
                    fact.datum_type
                )));
            }
            let s = match shape_to_concrete_i64(&fact.shape) {
                Some(s) => s,
                None => {
                    return Ok(ConcatAnalysis::Skip(format!(
                        "Concat input {slot} symbolic shape: {:?}",
                        fact.shape
                    )));
                }
            };
            raw_shapes.push(s);
            const_tensors.push(None);
        }
    }
    let rank = raw_shapes[0].len();
    if !(2..=4).contains(&rank) {
        return Ok(ConcatAnalysis::Skip(format!(
            "Concat input rank {rank} (only 2..=4 supported)"
        )));
    }
    if !raw_shapes.iter().all(|s| s.len() == rank) {
        return Ok(ConcatAnalysis::Skip(format!(
            "Concat inputs have differing ranks: {:?}",
            raw_shapes.iter().map(|s| s.len()).collect::<Vec<_>>()
        )));
    }

    // axis is in tract-side rank coords; shift to MLPackage-side rank-4.
    let pad = 4 - rank;
    let axis_ml = (concat.axis + pad) as i64;
    if !(0..4).contains(&axis_ml) {
        return Ok(ConcatAnalysis::Skip(format!(
            "Concat axis {} maps to MLPackage axis {axis_ml} (out of range)",
            concat.axis
        )));
    }

    // Pad each input to rank 4 by prepending 1s.
    let input_shapes: Vec<Vec<i64>> = raw_shapes
        .iter()
        .map(|s| {
            let mut padded = vec![1; pad];
            padded.extend_from_slice(s);
            padded
        })
        .collect();

    // Validate: non-axis dims must match across all inputs.
    let ref_shape = &input_shapes[0];
    for (i, s) in input_shapes.iter().enumerate().skip(1) {
        for (ax, (&r, &v)) in ref_shape.iter().zip(s.iter()).enumerate() {
            if (ax as i64) != axis_ml && r != v {
                return Ok(ConcatAnalysis::Skip(format!(
                    "Concat non-axis dim mismatch at axis {ax}: input 0 = {r}, input {i} = {v}"
                )));
            }
        }
    }

    // Output shape: same as ref_shape but with axis dim summed.
    let mut output_shape = ref_shape.clone();
    output_shape[axis_ml as usize] = input_shapes.iter().map(|s| s[axis_ml as usize]).sum();

    // Pair shapes with const-or-data into ConcatInput.
    let inputs: Vec<ConcatInput> = input_shapes
        .into_iter()
        .zip(const_tensors.into_iter())
        .map(|(shape, ct)| match ct {
            Some(t) => ConcatInput::Const { shape, tensor: t },
            None => ConcatInput::Data { shape },
        })
        .collect();

    Ok(ConcatAnalysis::Translatable(ConcatPlan {
        inputs,
        output_shape,
        axis: axis_ml,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

/// Emit MIL ops for a Concat. `input_data_names` has one entry per tract
/// input slot (in order): `Some(name)` for a Data input the caller wired
/// by name (must match `ConcatInput::Data` at that slot); `None` for a
/// Const input that emit handles inline (must match `ConcatInput::Const`).
pub fn emit_concat_mil(
    plan: &ConcatPlan,
    blob: &mut BlobBuilder,
    input_data_names: &[Option<&str>],
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    // axis as int32 const.
    let axis_const_name = format!("{output_name}_axis");
    let axis_const = op_const_immediate(
        &axis_const_name,
        tensor_type_scalar(DataType::Int32),
        tv_ints(vec![plan.axis as i32]),
    );

    // interleave const = false.
    let interleave_const_name = format!("{output_name}_interleave");
    let interleave_const = op_const_immediate(
        &interleave_const_name,
        tensor_type_scalar(DataType::Bool),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: vec![false],
            })),
        },
    );

    // Defensive input-rank normalization. analyse_concat pads each input's
    // shape to rank-4 (and shifts axis correspondingly), but the upstream
    // MIL ops may emit at tract's own rank — e.g. a Reshape that drops to
    // rank-2 [C, H*W] in YOLOv8's head. Without explicit reshapes here, MIL
    // sees the rank-2 inputs but our axis is in the rank-4 frame and the
    // concat fails: "Axis (3) must be within range [-2, 2) for ios17.concat".
    //
    // Emit one MIL `reshape` per input bringing it to plan.input_shapes[i]
    // (rank-4). When the upstream is already rank-4 with the matching shape
    // this reshape is metadata-only — no compute cost. Same idea as Conv's
    // x_rank4 reshape; see ops/conv.rs.
    if input_data_names.len() != plan.inputs.len() {
        bail!(
            "Concat::emit: caller passed {} data-name slots but plan has {} inputs",
            input_data_names.len(),
            plan.inputs.len()
        );
    }

    let mut prelude: Vec<mil::Operation> = vec![axis_const, interleave_const];
    let mut concat_value_names: Vec<String> = Vec::with_capacity(plan.inputs.len());
    for (idx, (plan_input, data_name_opt)) in
        plan.inputs.iter().zip(input_data_names.iter()).enumerate()
    {
        let in_shape = match plan_input {
            ConcatInput::Data { shape } | ConcatInput::Const { shape, .. } => shape,
        };

        // Materialise the upstream value as a rank-4 named tensor. For a
        // Data input the caller passed us the upstream name; for a Const
        // input we emit `op_const_blob` writing the bytes to the shared
        // weight blob. Either way, we then defensively reshape to the
        // plan's rank-4 shape (no-op for Const since op_const_blob declared
        // the correct shape; metadata-only for Data when ranks already
        // match; corrects rank-2/rank-3 → rank-4 promotion otherwise).
        let upstream_name: String = match (plan_input, data_name_opt) {
            (ConcatInput::Data { .. }, Some(n)) => n.to_string(),
            (ConcatInput::Data { .. }, None) => {
                bail!("Concat::emit: slot {idx} is Data but caller passed no name (fusion bug)")
            }
            (ConcatInput::Const { shape, tensor }, None) => {
                let const_name = format!("{output_name}_in{idx}_const");
                let const_ty = tensor_type(DataType::Float16, shape);
                let offset = blob.add(BlobDataType::Float16, tensor.as_bytes());
                let const_op =
                    op_const_blob(&const_name, const_ty, mlpackage::WEIGHT_BLOB_PATH, offset);
                prelude.push(const_op);
                const_name
            }
            (ConcatInput::Const { .. }, Some(_)) => {
                bail!("Concat::emit: slot {idx} is Const but caller passed a name (fusion bug)")
            }
        };

        let reshaped_name = format!("{output_name}_in{idx}_rank4");
        let shape_const_name = format!("{output_name}_in{idx}_shape");
        let shape_const = op_const_immediate(
            &shape_const_name,
            tensor_type(DataType::Int32, &[in_shape.len() as i64]),
            tv_ints(in_shape.iter().map(|&v| v as i32).collect()),
        );
        let reshape_in_ty = tensor_type(DataType::Float16, in_shape);
        let mut reshape_inputs: HashMap<String, mil::Argument> = HashMap::new();
        reshape_inputs.insert("x".into(), arg_name(&upstream_name));
        reshape_inputs.insert("shape".into(), arg_name(&shape_const_name));
        let reshape_op = mil::Operation {
            r#type: "reshape".into(),
            inputs: reshape_inputs,
            outputs: vec![mil::NamedValueType {
                name: reshaped_name.clone(),
                r#type: Some(reshape_in_ty),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        };
        prelude.push(shape_const);
        prelude.push(reshape_op);
        concat_value_names.push(reshaped_name);
    }
    // Need &str refs for arg_names.
    let concat_refs: Vec<&str> = concat_value_names.iter().map(|s| s.as_str()).collect();

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("values".into(), arg_names(&concat_refs));
    inputs.insert("axis".into(), arg_name(&axis_const_name));
    inputs.insert("interleave".into(), arg_name(&interleave_const_name));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let concat_op = mil::Operation {
        r#type: "concat".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };
    prelude.push(concat_op);

    Ok(prelude)
}

//! tract `RmsNorm` → MIL primitive chain (`square + reduce_mean + add(eps)
//! + rsqrt + mul`).
//!
//! `RmsNorm(x) = x * rsqrt(mean(x², axes) + eps)`. Unlike LayerNorm, no
//! mean-subtraction (so it normalises by RMS rather than std). MIL doesn't
//! have a first-class `rms_norm` op (as of Core ML 7), so we lower to the
//! 5-op chain above.
//!
//! Two reasons to ship a standalone translator:
//!
//!   * **Modern LLM transformers** (LLaMA / Mistral / Qwen) replace
//!     LayerNorm with RMSNorm directly — without this translator they'd
//!     hit a CPU residual on every block.
//!
//!   * **Subgraph cohesion for ONNX LayerNorm**. tract's `into_decluttered`
//!     turns the LayerNorm decomposition into `Mean → Sub → RmsNorm →
//!     Mul(γ) → Add(β)`. If RmsNorm isn't translatable, it acts as a CPU
//!     barrier that splits the LayerNorm chain into THREE subgraphs (the
//!     pre-mean part, the post-mean γβ part) and the multi-node LayerNorm
//!     fold (in [`super::layer_norm`]) never gets a chance to run. Making
//!     RmsNorm translatable keeps the whole chain in one subgraph, and the
//!     fold pre-scan then collapses it to a single MIL `layer_norm` op.
//!
//! ## Limitations
//!
//!   * **F16 only**.
//!   * **Single axis** at `RmsNorm.axis`.

use std::collections::HashMap;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::nn::RmsNorm;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;
use crate::proto::core_ml::specification::mil_spec::tensor_value::RepeatedBytes;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum RmsNormAnalysis {
    Translatable(RmsNormPlan),
    Skip(String),
}

pub struct RmsNormPlan {
    /// MLPackage-side input shape (rank as-is).
    pub input_shape: Vec<i64>,
    /// Reduce axis (single).
    pub axis: i32,
    /// Epsilon as f32 (will be re-cast to F16 when emitted).
    pub epsilon: f32,
    pub output_fact: TypedFact,
}

pub fn analyse_rms_norm(model: &TypedModel, node: &TypedNode) -> Result<RmsNormAnalysis> {
    let Some(rn) = node.op_as::<RmsNorm>() else {
        return Ok(RmsNormAnalysis::Skip("not a RmsNorm".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(RmsNormAnalysis::Skip(format!(
            "RmsNorm has {} inputs (need 1)",
            node.inputs.len()
        )));
    }
    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(RmsNormAnalysis::Skip(format!(
            "RmsNorm input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let raw_in = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(RmsNormAnalysis::Skip(format!(
                "RmsNorm input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if !(2..=5).contains(&raw_in.len()) {
        return Ok(RmsNormAnalysis::Skip(format!(
            "RmsNorm input rank {} (only 2..=5 supported)",
            raw_in.len()
        )));
    }

    // Same rank-padding correction as reduce.rs: upstream translators
    // (Conv, BinOp, Reduce, etc.) normalise rank-3 CHW tensors to rank-4
    // by prepending N=1. The MLPackage tensor flowing in here may already
    // be padded, so we shift the axis to match.
    let pad = 4_usize.saturating_sub(raw_in.len());
    let input_shape = super::rank::pad_to_rank_4(&raw_in);
    let axis_padded = rn.axis as i32 + pad as i32;
    if axis_padded < 0 || (axis_padded as usize) >= input_shape.len() {
        return Ok(RmsNormAnalysis::Skip(format!(
            "RmsNorm axis {} out of range for rank {} (after rank-pad)",
            axis_padded,
            input_shape.len()
        )));
    }
    let epsilon = match rn.eps.datum_type() {
        DatumType::F16 => unsafe { rn.eps.as_slice_unchecked::<f16>()[0].to_f32() },
        DatumType::F32 => unsafe { rn.eps.as_slice_unchecked::<f32>()[0] },
        DatumType::F64 => unsafe { rn.eps.as_slice_unchecked::<f64>()[0] as f32 },
        other => {
            return Ok(RmsNormAnalysis::Skip(format!("RmsNorm eps dtype {other:?} unsupported")));
        }
    };

    Ok(RmsNormAnalysis::Translatable(RmsNormPlan {
        input_shape,
        axis: axis_padded,
        epsilon,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_rms_norm_mil(
    plan: &RmsNormPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;
    let mut ops: Vec<mil::Operation> = Vec::new();

    // Defensive input-rank normalization. Same pattern as Conv / Concat /
    // Slice / Activation: analyse_rms_norm pads `plan.input_shape` to
    // rank-4 (via `pad_to_rank_4`), but the upstream MIL value may be at a
    // different rank — e.g. a Gather output that emits at tract's natural
    // rank-3 [batch, seq, hidden]. MIL `square` is element-wise so its
    // output rank == input rank; without aligning, our rank-4 declaration
    // mismatches MIL's rank-3 inference and the compile aborts with
    // "Output '0' has unexpected type 'ios17.square'. Expected rank-3; got
    // rank-4" (or vice versa). Surfaced on SmolLM2 once Gather started
    // translating the embedding-lookup output. Emit a single leading
    // reshape that materialises the input at plan.input_shape (rank-4);
    // metadata-only when ranks already match.
    let input_rank4_name = format!("{p}_x_rank{}", plan.input_shape.len());
    let input_shape_n = format!("{p}_x_shape");
    let input_shape_op = op_const_immediate(
        &input_shape_n,
        tensor_type(DataType::Int32, &[plan.input_shape.len() as i64]),
        tv_ints(plan.input_shape.iter().map(|&v| v as i32).collect()),
    );
    let input_rank4_ty = tensor_type(DataType::Float16, &plan.input_shape);
    ops.push(input_shape_op);
    ops.push(mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(input_name)),
            ("shape".into(), arg_name(&input_shape_n)),
        ]),
        outputs: vec![mil::NamedValueType {
            name: input_rank4_name.clone(),
            r#type: Some(input_rank4_ty.clone()),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // 1. square(x).
    let sq_n = format!("{p}_square");
    let sq_ty = tensor_type(DataType::Float16, &plan.input_shape);
    ops.push(mil::Operation {
        r#type: "square".into(),
        inputs: HashMap::from([("x".into(), arg_name(&input_rank4_name))]),
        outputs: vec![mil::NamedValueType { name: sq_n.clone(), r#type: Some(sq_ty.clone()) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // 2. reduce_mean(x², axis, keep_dims=true).
    let axes_n = format!("{p}_axes");
    let keep_n = format!("{p}_keep");
    let axes_ty = tensor_type(DataType::Int32, &[1]);
    ops.push(op_const_immediate(&axes_n, axes_ty, tv_ints(vec![plan.axis])));
    ops.push(op_const_immediate(
        &keep_n,
        tensor_type_scalar(DataType::Bool),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: vec![true],
            })),
        },
    ));
    let mut reduced_shape = plan.input_shape.clone();
    reduced_shape[plan.axis as usize] = 1;
    let mean_n = format!("{p}_mean");
    let reduced_ty = tensor_type(DataType::Float16, &reduced_shape);
    ops.push(mil::Operation {
        r#type: "reduce_mean".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&sq_n)),
            ("axes".into(), arg_name(&axes_n)),
            ("keep_dims".into(), arg_name(&keep_n)),
        ]),
        outputs: vec![mil::NamedValueType {
            name: mean_n.clone(),
            r#type: Some(reduced_ty.clone()),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // 3. add(mean, eps). Eps as F16 scalar.
    let eps_n = format!("{p}_eps");
    let eps_f16 = f16::from_f32(plan.epsilon);
    ops.push(op_const_immediate(
        &eps_n,
        tensor_type_scalar(DataType::Float16),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bytes(RepeatedBytes {
                values: eps_f16.to_le_bytes().to_vec(),
            })),
        },
    ));
    let var_eps_n = format!("{p}_var_eps");
    ops.push(mil::Operation {
        r#type: "add".into(),
        inputs: HashMap::from([("x".into(), arg_name(&mean_n)), ("y".into(), arg_name(&eps_n))]),
        outputs: vec![mil::NamedValueType {
            name: var_eps_n.clone(),
            r#type: Some(reduced_ty.clone()),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // 4. rsqrt(_) — MIL rsqrt takes its own epsilon arg; we pass a tiny one
    //    since we already added eps in step 3.
    let inner_eps_n = format!("{p}_rsqrt_eps");
    ops.push(op_const_immediate(
        &inner_eps_n,
        tensor_type_scalar(DataType::Float32),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Floats(mil::tensor_value::RepeatedFloats {
                values: vec![1e-12],
            })),
        },
    ));
    let inv_std_n = format!("{p}_inv_std");
    ops.push(mil::Operation {
        r#type: "rsqrt".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&var_eps_n)),
            ("epsilon".into(), arg_name(&inner_eps_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: inv_std_n.clone(), r#type: Some(reduced_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // 5. mul(x_rank4, inv_std). Use the rank-aligned input (not raw
    // input_name) so MIL's broadcast rules see consistent ranks.
    let out_ty = tensor_type(DataType::Float16, &plan.input_shape);
    ops.push(mil::Operation {
        r#type: "mul".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&input_rank4_name)),
            ("y".into(), arg_name(&inv_std_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    Ok(ops)
}

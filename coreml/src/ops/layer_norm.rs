//! Multi-node LayerNorm pattern → MIL `layer_norm` translator.
//!
//! ONNX `LayerNormalization` is decomposed by tract's `into_decluttered`
//! into a 6-op chain (much shorter than the raw 11-op decomposition,
//! because tract has a first-class `RmsNorm` op that absorbs the variance
//! + epsilon + rsqrt + multiply step):
//!
//! ```text
//!   mean_sum = ReduceSum(x, axes)             // axes typically = [last]
//!   mean     = Mul(mean_sum, 1/N const)       // Div folded to Mul(_, 1/N)
//!   d        = Sub(x, mean)
//!   rms_d    = RmsNorm(d)                     // = d * rsqrt(mean(d²) + eps)
//!   scaled   = Mul(rms_d, gamma const)
//!   y        = Add(scaled, beta const)        // optional
//! ```
//!
//! Apple's MIL has `layer_norm(x, axes, gamma, beta, epsilon)` on ANE's
//! hardware fastpath. Critical for transformers — every block has at least
//! one LN.
//!
//! ## Detection
//!
//! Anchored at the final Add (with-bias) or final Mul (no-bias). Walks
//! back through 4–5 ops verifying the chain. Epsilon comes from the
//! `RmsNorm.eps` field, not from a separate `Add(var, eps)` since declutter
//! has already absorbed that into RmsNorm.
//!
//! ## Limitations
//!
//!   * **Single normalisation axis** at the trailing position (axis = -1).
//!     Multi-axis LN (vision transformer with 2D normalisation) deferred.
//!   * **F16 only**.
//!   * If the chain doesn't match (e.g. extra Cast slips through, or
//!     declutter behaves differently for some shape), the detector
//!     returns `None` and the chain falls back to per-op translation —
//!     correctness is preserved either way; only the consolidation
//!     benefit is lost (and we'd lose ANE's fast `layer_norm` op).

use std::collections::HashMap;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_core::ops::math::{Add, Mul, Sub};
use tract_core::ops::nn::{Reduce, Reducer, RmsNorm};

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;
use crate::proto::core_ml::specification::mil_spec::tensor_value::RepeatedBytes;

use super::{const_tensor, shape_to_concrete_i64};

pub struct LayerNormPlan {
    /// External data input feeding the chain (the `x` arg of `layer_norm`).
    pub data_input: OutletId,
    /// MLPackage-side input shape.
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (same as input).
    pub output_shape: Vec<i64>,
    /// Normalisation axes — single trailing axis in MLPackage rank coords.
    pub axes: Vec<i32>,
    /// Per-feature dim used for gamma/beta length (= input_shape[axes[-1]]).
    pub norm_size: usize,
    /// Gamma (scale) tensor, length-`norm_size`, F16.
    pub gamma: Tensor,
    /// Beta (bias) tensor, length-`norm_size`, F16. Zeros if no bias Add.
    pub beta: Tensor,
    /// Epsilon from the `RmsNorm` op.
    pub epsilon: f32,
    pub output_fact: TypedFact,
    /// Node IDs of absorbed chain members (excluding the anchor). Caller
    /// skips emitting these.
    pub absorbed: Vec<usize>,
}

fn is_binop<T: 'static + BinMiniOp>(node: &TypedNode) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.as_ref().downcast_ref::<T>().is_some())
}

/// Try to detect a LayerNorm chain anchored at `node`.
pub fn detect_layer_norm(
    model: &TypedModel,
    node: &TypedNode,
    in_subgraph: &dyn Fn(usize) -> bool,
) -> Option<LayerNormPlan> {
    // ---- Anchor: with-bias case is Add(scaled, beta_const). Tuple's
    // first slot is unused in the fold logic but kept so the matcher reads
    // top-down (we only need to know whether `beta_const_outlet` is Some).
    let (_with_bias, scaled_node, beta_const_outlet) =
        if is_binop::<Add>(node) && node.inputs.len() == 2 {
            let a_const = const_tensor(model, node.inputs[0]).ok().flatten();
            let b_const = const_tensor(model, node.inputs[1]).ok().flatten();
            let lhs = &model.nodes[node.inputs[0].node];
            let rhs = &model.nodes[node.inputs[1].node];
            if a_const.is_none() && b_const.is_some() && is_binop::<Mul>(lhs) {
                (true, lhs, Some(node.inputs[1]))
            } else if b_const.is_none() && a_const.is_some() && is_binop::<Mul>(rhs) {
                (true, rhs, Some(node.inputs[0]))
            } else {
                return None;
            }
        } else if is_binop::<Mul>(node) && node.inputs.len() == 2 {
            (false, node, None)
        } else {
            return None;
        };

    // scaled = Mul(rms_d, gamma_const).
    let (rms_node, gamma_const_outlet) = {
        let a_const = const_tensor(model, scaled_node.inputs[0]).ok().flatten();
        let b_const = const_tensor(model, scaled_node.inputs[1]).ok().flatten();
        if a_const.is_none() && b_const.is_some() {
            (&model.nodes[scaled_node.inputs[0].node], scaled_node.inputs[1])
        } else if a_const.is_some() && b_const.is_none() {
            (&model.nodes[scaled_node.inputs[1].node], scaled_node.inputs[0])
        } else {
            return None;
        }
    };

    // rms_d = RmsNorm(d, eps=...).
    let rms = rms_node.op_as::<RmsNorm>()?;
    if rms_node.inputs.len() != 1 {
        return None;
    }
    let rms_axis = rms.axis;
    let eps = match rms.eps.datum_type() {
        DatumType::F16 => unsafe { rms.eps.as_slice_unchecked::<f16>()[0].to_f32() },
        DatumType::F32 => unsafe { rms.eps.as_slice_unchecked::<f32>()[0] },
        DatumType::F64 => unsafe { rms.eps.as_slice_unchecked::<f64>()[0] as f32 },
        _ => return None,
    };

    let d_node = &model.nodes[rms_node.inputs[0].node];
    // d = Sub(x, mean).
    if !is_binop::<Sub>(d_node) || d_node.inputs.len() != 2 {
        return None;
    }
    let x_outlet = d_node.inputs[0];
    let mean_outlet = d_node.inputs[1];

    // mean is either:
    //   (a) Mul(mean_sum, 1/N const), where mean_sum = ReduceSum(x, axes), or
    //   (b) the rank-preserving "ReduceMean" — declutter may keep this as
    //       a Reduce<Sum> + Mul, OR swap the order, OR use a ReduceMean
    //       reducer if tract gains one (currently doesn't).
    let mean_node = &model.nodes[mean_outlet.node];
    let mean_sum_node_id = if is_binop::<Mul>(mean_node) && mean_node.inputs.len() == 2 {
        let a_const = const_tensor(model, mean_node.inputs[0]).ok().flatten();
        let b_const = const_tensor(model, mean_node.inputs[1]).ok().flatten();
        if a_const.is_none() && b_const.is_some() {
            mean_node.inputs[0].node
        } else if a_const.is_some() && b_const.is_none() {
            mean_node.inputs[1].node
        } else {
            return None;
        }
    } else {
        return None;
    };
    let mean_sum_node = &model.nodes[mean_sum_node_id];
    let mean_sum_red = mean_sum_node.op_as::<Reduce>()?;
    if !matches!(mean_sum_red.reducer, Reducer::Sum) || mean_sum_node.inputs.len() != 1 {
        return None;
    }
    if mean_sum_node.inputs[0] != x_outlet {
        return None;
    }
    if mean_sum_red.axes.len() != 1 || mean_sum_red.axes[0] != rms_axis {
        return None;
    }

    // ---- Validate input data fact + axes layout.
    let x_fact = model.outlet_fact(x_outlet).ok()?;
    if x_fact.datum_type != DatumType::F16 {
        return None;
    }
    let raw_in = shape_to_concrete_i64(&x_fact.shape)?;
    if !(2..=5).contains(&raw_in.len()) {
        return None;
    }
    if rms_axis != raw_in.len() - 1 {
        return None;
    }
    // Rank-padding correction (same as reduce.rs / rms_norm.rs): upstream
    // translators normalise to rank 4. The chain intermediates we collapse
    // have already been padded, so the consolidated MIL `layer_norm` must
    // declare its shape in the padded rank.
    let pad = 4_usize.saturating_sub(raw_in.len());
    let input_shape = super::rank::pad_to_rank_4(&raw_in);
    let axis = rms_axis as i32 + pad as i32;
    let norm_size = input_shape[axis as usize] as usize;

    // ---- Extract gamma (and beta if with_bias).
    let gamma_const = const_tensor(model, gamma_const_outlet).ok().flatten()?.into_owned();
    if gamma_const.datum_type() != DatumType::F16 || gamma_const.len() != norm_size {
        return None;
    }
    let gamma_flat = unsafe {
        Tensor::from_raw_dt(gamma_const.datum_type(), &[norm_size], gamma_const.as_bytes()).ok()?
    };

    let beta_flat = if let Some(beta_outlet) = beta_const_outlet {
        let beta_const = const_tensor(model, beta_outlet).ok().flatten()?.into_owned();
        if beta_const.datum_type() != DatumType::F16 || beta_const.len() != norm_size {
            return None;
        }
        unsafe {
            Tensor::from_raw_dt(beta_const.datum_type(), &[norm_size], beta_const.as_bytes())
                .ok()?
        }
    } else {
        let zeros: Vec<f16> = vec![f16::ZERO; norm_size];
        Tensor::from_shape::<f16>(&[norm_size], &zeros).ok()?
    };

    // ---- Verify chain membership in subgraph + no external consumers.
    // Members in walk order (anchor's predecessors, deepest first).
    let mut absorbed: Vec<usize> =
        vec![scaled_node.id, rms_node.id, d_node.id, mean_node.id, mean_sum_node.id];
    absorbed.retain(|&id| id != node.id);

    for &m in &absorbed {
        if !in_subgraph(m) {
            return None;
        }
    }
    let chain_set: std::collections::HashSet<usize> =
        absorbed.iter().copied().chain(std::iter::once(node.id)).collect();
    for &m in &absorbed {
        let m_outlet = OutletId::new(m, 0);
        for other in &model.nodes {
            if !chain_set.contains(&other.id) && other.inputs.contains(&m_outlet) {
                return None;
            }
        }
    }

    Some(LayerNormPlan {
        data_input: x_outlet,
        input_shape: input_shape.clone(),
        output_shape: input_shape,
        axes: vec![axis],
        norm_size,
        gamma: gamma_flat,
        beta: beta_flat,
        epsilon: eps,
        output_fact: node.outputs[0].fact.clone(),
        absorbed,
    })
}

pub fn emit_layer_norm_mil(
    plan: &LayerNormPlan,
    blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;

    let gamma_off = blob.add(BlobDataType::Float16, plan.gamma.as_bytes());
    let beta_off = blob.add(BlobDataType::Float16, plan.beta.as_bytes());

    let g_ty = tensor_type(DataType::Float16, &[plan.norm_size as i64]);
    let gamma_n = format!("{p}_gamma");
    let beta_n = format!("{p}_beta");
    let gamma_op = op_const_blob(&gamma_n, g_ty.clone(), mlpackage::WEIGHT_BLOB_PATH, gamma_off);
    let beta_op = op_const_blob(&beta_n, g_ty, mlpackage::WEIGHT_BLOB_PATH, beta_off);

    // Epsilon as F16 scalar bytes (gamma/beta/epsilon must share dtype).
    let eps_f16 = f16::from_f32(plan.epsilon);
    let eps_n = format!("{p}_epsilon");
    let eps_op = op_const_immediate(
        &eps_n,
        tensor_type_scalar(DataType::Float16),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bytes(RepeatedBytes {
                values: eps_f16.to_le_bytes().to_vec(),
            })),
        },
    );

    let axes_n = format!("{p}_axes");
    let axes_ty = tensor_type(DataType::Int32, &[plan.axes.len() as i64]);
    let axes_op = op_const_immediate(&axes_n, axes_ty, tv_ints(plan.axes.clone()));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let ln_op = mil::Operation {
        r#type: "layer_norm".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(input_name)),
            ("axes".into(), arg_name(&axes_n)),
            ("gamma".into(), arg_name(&gamma_n)),
            ("beta".into(), arg_name(&beta_n)),
            ("epsilon".into(), arg_name(&eps_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![gamma_op, beta_op, eps_op, axes_op, ln_op])
}

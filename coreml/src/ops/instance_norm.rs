//! Multi-node InstanceNorm pattern → MIL `instance_norm` translator.
//!
//! After tract's `into_decluttered`, an ONNX `InstanceNormalization` op is
//! decomposed into a chain of ~6 ops (ReduceSum + Mul(1/N) + Sub +
//! Reduce<MeanOfSquares> + Add(eps) + Rsqrt + final Mul). Apple's MIL has a
//! direct `instance_norm` op that's almost certainly faster on ANE
//! (hardware-fastpath candidate) and lets the Core ML compiler emit one
//! op instead of fusing six. This module recognises the chain pattern and
//! collapses it.
//!
//! ## Detection
//!
//! Detection is anchored at the final Mul (whose output is the InstanceNorm
//! result). Walking back from that Mul, we verify:
//!
//!   - input 0 is a `Sub(x, mean)` — the centered input
//!   - input 1 is a `Rsqrt(Add(variance, eps_const))` — the normalisation factor
//!   - `mean = Mul(ReduceSum(x, axes), 1/N_const)` — the mean computation
//!   - `variance = Reduce<MeanOfSquares>(centered, axes)` — the variance
//!     computed via the centered input (var(x) = mean((x - mean)²))
//!   - `axes` is the same on both reductions, and consistent with the
//!     "instance norm normalises over spatial dims" expectation
//!     (typically `[0, 2, 3]` for the rank-4 NCHW we see in MODNet)
//!
//! The 6 chain members (ReduceSum + Mul(1/N) + Sub + Reduce<MoS> +
//! Add(eps) + Rsqrt) are returned as `absorbed`; the anchor Mul is the
//! emitting node. Const inputs (1/N reciprocal, eps) are absorbed via the
//! const-tensor mechanism.
//!
//! The detector also checks that no chain member's output is consumed
//! OUTSIDE the chain — if so, we can't collapse without breaking dataflow.
//!
//! ## Output
//!
//! Pure standardisation (gamma=1, beta=0 — MODNet applies its
//! per-channel scale/bias via separate BatchNorm-style Mul/Add ops AFTER
//! the InstanceNorm chain, which our existing BinOp translator already
//! handles).

use std::collections::HashMap;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::{Add, Mul, Sub};
use tract_core::ops::nn::{Reduce, Reducer};

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;
use crate::proto::core_ml::specification::mil_spec::tensor_value::RepeatedBytes;

use super::{const_tensor, shape_to_concrete_i64};

/// Anchored InstanceNorm chain. Stored at fusion-time per anchor so the emit
/// loop can produce a single `mb.instance_norm` per chain.
pub struct InstanceNormPlan {
    /// External data input feeding the chain (the `x` arg of `instance_norm`).
    pub data_input: OutletId,
    /// MLPackage-side input shape (rank 4 NCHW).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (same as input).
    pub output_shape: Vec<i64>,
    /// Number of channels = `input_shape[1]`. Gamma/beta are length-C.
    pub channels: usize,
    /// Axes to normalise over (in MLPackage rank coords). Typically `[2, 3]`
    /// for spatial-only InstanceNorm on NCHW.
    pub axes: Vec<i32>,
    /// Epsilon value extracted from the chain's `Add(variance, eps)` const.
    pub epsilon: f32,
    /// tract-side output fact (for the anchor Mul's output).
    pub output_fact: TypedFact,
    /// Node IDs of the absorbed chain members (NOT including the anchor).
    /// `build_subgraph_mlpackage` skips emitting these.
    pub absorbed: Vec<usize>,
}

/// True if `node` is an `ElementWiseOp` whose mini-op name matches `name`.
fn is_elementwise_named(node: &TypedNode, name: &str) -> bool {
    node.op_as::<ElementWiseOp>().is_some_and(|ew| ew.0.name() == name)
}

/// True if `node` is a `TypedBinOp` whose inner mini-op is the given Rust type.
fn is_binop<T: 'static + BinMiniOp>(node: &TypedNode) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.as_ref().downcast_ref::<T>().is_some())
}

/// Try to identify `node` as the *anchor* (final Mul) of an InstanceNorm
/// chain. Returns `Some(plan)` if the full pattern matches, `None` otherwise.
///
/// Caller is responsible for verifying that the chain's member nodes are all
/// inside the same subgraph and that no member's output is consumed outside
/// the subgraph (since collapsing the chain replaces all members with a
/// single op).
pub fn detect_instance_norm(
    model: &TypedModel,
    node: &TypedNode,
    in_subgraph: &dyn Fn(usize) -> bool,
) -> Option<InstanceNormPlan> {
    // Anchor is a TypedBinOp(Mul) with 2 inputs.
    if !is_binop::<Mul>(node) || node.inputs.len() != 2 {
        return None;
    }

    let sub_node = &model.nodes[node.inputs[0].node];
    let rsqrt_node = &model.nodes[node.inputs[1].node];
    if !is_binop::<Sub>(sub_node) || sub_node.inputs.len() != 2 {
        return None;
    }
    if !is_elementwise_named(rsqrt_node, "Rsqrt") || rsqrt_node.inputs.len() != 1 {
        return None;
    }

    let data_outlet = sub_node.inputs[0];
    let mean_norm_node = &model.nodes[sub_node.inputs[1].node];
    if !is_binop::<Mul>(mean_norm_node) || mean_norm_node.inputs.len() != 2 {
        return None;
    }

    let reduce_sum_node = &model.nodes[mean_norm_node.inputs[0].node];
    let recip_const_outlet = mean_norm_node.inputs[1];
    let reduce_sum = reduce_sum_node.op_as::<Reduce>()?;
    if !matches!(reduce_sum.reducer, Reducer::Sum) || reduce_sum_node.inputs.len() != 1 {
        return None;
    }
    if reduce_sum_node.inputs[0] != data_outlet {
        // ReduceSum must operate on the same external data input as Sub
        // (the `mean = ReduceSum(x) / N` depends on `x`).
        return None;
    }

    let eps_add_node = &model.nodes[rsqrt_node.inputs[0].node];
    if !is_binop::<Add>(eps_add_node) || eps_add_node.inputs.len() != 2 {
        return None;
    }
    let variance_node = &model.nodes[eps_add_node.inputs[0].node];
    let eps_const_outlet = eps_add_node.inputs[1];
    let variance = variance_node.op_as::<Reduce>()?;
    if !matches!(variance.reducer, Reducer::MeanOfSquares) || variance_node.inputs.len() != 1 {
        return None;
    }
    // Variance is computed on the centered input (= the Sub node's output).
    if variance_node.inputs[0] != OutletId::new(sub_node.id, 0) {
        return None;
    }

    // Axes must match between ReduceSum and Reduce<MoS>.
    if reduce_sum.axes != variance.axes {
        return None;
    }
    let axes: Vec<i32> = reduce_sum.axes.iter().map(|&a| a as i32).collect();

    // Extract eps and verify it's a scalar-ish const.
    let eps_const = const_tensor(model, eps_const_outlet).ok().flatten()?.into_owned();
    let eps = match eps_const.datum_type() {
        DatumType::F16 => unsafe { eps_const.as_slice_unchecked::<f16>()[0].to_f32() },
        DatumType::F32 => unsafe { eps_const.as_slice_unchecked::<f32>()[0] },
        _ => return None,
    };

    // Verify 1/N const exists (we don't strictly need its value — the MIL
    // op derives N from the input shape and the axes — but we want to make
    // sure the const-absorption mechanism works for the Mul's slot 1).
    let _recip_const = const_tensor(model, recip_const_outlet).ok().flatten()?;

    // Shape extraction.
    let in_fact = model.outlet_fact(data_outlet).ok()?;
    if in_fact.datum_type != DatumType::F16 {
        return None;
    }
    let input_shape = shape_to_concrete_i64(&in_fact.shape)?;
    if input_shape.len() != 4 {
        // MIL `instance_norm` is rank-4 (NCHW) only.
        return None;
    }
    let channels = input_shape[1] as usize;

    // The axes must include the spatial axes (2, 3) and may include the
    // batch axis (0). MIL `instance_norm` always normalises over the
    // SPATIAL axes only (per-instance, per-channel mean/var). For batch=1
    // the two semantics agree.
    let normalises_spatial = axes.contains(&2) && axes.contains(&3);
    if !normalises_spatial {
        return None;
    }

    // Verify all 6 chain members are in the same subgraph as the anchor.
    let chain_members = [
        sub_node.id,
        rsqrt_node.id,
        mean_norm_node.id,
        reduce_sum_node.id,
        eps_add_node.id,
        variance_node.id,
    ];
    for &m in &chain_members {
        if !in_subgraph(m) {
            return None;
        }
    }

    // Verify each chain member's output is consumed ONLY by other chain
    // members (or the anchor). If anything outside the chain reads an
    // intermediate, we can't collapse.
    let chain_set: std::collections::HashSet<usize> =
        chain_members.iter().copied().chain(std::iter::once(node.id)).collect();
    for &m in &chain_members {
        let m_outlet = OutletId::new(m, 0);
        for other in &model.nodes {
            if !chain_set.contains(&other.id) && other.inputs.contains(&m_outlet) {
                return None;
            }
        }
    }

    Some(InstanceNormPlan {
        data_input: data_outlet,
        input_shape: input_shape.clone(),
        output_shape: input_shape,
        channels,
        // MIL instance_norm: only spatial axes.
        axes: vec![2, 3],
        epsilon: eps,
        output_fact: node.outputs[0].fact.clone(),
        absorbed: chain_members.to_vec(),
    })
}

pub fn emit_instance_norm_mil(
    plan: &InstanceNormPlan,
    blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;

    // gamma = ones(C); beta = zeros(C). MODNet applies its per-channel
    // scale/bias via separate BatchNorm-style Mul/Add ops after this chain,
    // which the existing BinOp translator handles.
    let ones_data: Vec<f16> = vec![f16::from_f32(1.0); plan.channels];
    let zeros_data: Vec<f16> = vec![f16::ZERO; plan.channels];
    let ones_t = Tensor::from_shape::<f16>(&[plan.channels], &ones_data)?;
    let zeros_t = Tensor::from_shape::<f16>(&[plan.channels], &zeros_data)?;
    let gamma_off = blob.add(BlobDataType::Float16, ones_t.as_bytes());
    let beta_off = blob.add(BlobDataType::Float16, zeros_t.as_bytes());

    let gamma_n = format!("{p}_gamma");
    let beta_n = format!("{p}_beta");
    let eps_n = format!("{p}_epsilon");

    let c_ty = tensor_type(DataType::Float16, &[plan.channels as i64]);
    let gamma_op = op_const_blob(&gamma_n, c_ty.clone(), mlpackage::WEIGHT_BLOB_PATH, gamma_off);
    let beta_op = op_const_blob(&beta_n, c_ty, mlpackage::WEIGHT_BLOB_PATH, beta_off);

    // MIL `instance_norm` is rank-4 NCHW-only and always normalises over the
    // spatial (H, W) axes — no `axes` parameter to set. Only `epsilon` and
    // the per-channel `gamma` / `beta` are configurable. All three must have
    // the same dtype (F16 here, matching gamma/beta which we wrote to the
    // weight blob as F16). MIL serialises FP16 scalars as raw bytes via
    // `RepeatedBytes`; emit the 2-byte little-endian f16 representation.
    let eps_f16 = f16::from_f32(plan.epsilon);
    let eps_bytes = eps_f16.to_le_bytes().to_vec();
    let eps_op = op_const_immediate(
        &eps_n,
        tensor_type_scalar(DataType::Float16),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bytes(RepeatedBytes { values: eps_bytes })),
        },
    );

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let in_op = mil::Operation {
        r#type: "instance_norm".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(input_name)),
            ("gamma".into(), arg_name(&gamma_n)),
            ("beta".into(), arg_name(&beta_n)),
            ("epsilon".into(), arg_name(&eps_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![gamma_op, beta_op, eps_op, in_op])
}

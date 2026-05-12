//! tract `EinSum` (matmul-shaped) → MIL `matmul` translator.
//!
//! Beyond the 1×1-conv-shaped einsums handled in [`super::einsum`], tract
//! lowers PyTorch dense layers (e.g. SE blocks, attention QKV projections)
//! into matmul-shaped einsums. The two patterns we cover here:
//!
//!   * `mkba,kn->n` — shape `[M=1, K, B=1, A=1] @ [K, N] → [N]`
//!     (classifier head dense layer feeding a 1D output)
//!   * `k,kn->mnab` — shape `[K] @ [K, N] → [M=1, N, A=1, B=1]`
//!     (inverse: 1D input projects to a rank-4 feature map of 1×1 spatial)
//!
//! Both reduce a single shared axis K and are equivalent to a single MIL
//! `mb.matmul(x, weight)` with reshape ops to bring the input/output into
//! the standard `[..., M, K] @ [..., K, N] → [..., M, N]` form. Const weight
//! is absorbed into the MILBlob v2 weight file (same path Conv uses).
//!
//! Tract emits these for SE blocks in MobileNet-class models when the dense
//! layers don't get folded into 1×1 convs by `into_decluttered`. They appear
//! in MODNet's classifier head.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::const_tensor;

#[allow(clippy::large_enum_variant)]
pub enum MatMulAnalysis {
    Translatable(MatMulPlan),
    Skip(String),
}

pub struct MatMulPlan {
    /// MLPackage-side input shape (the actual rank tract feeds in — may be 1
    /// for `k,kn->mnab` or 4 for `mkba,kn->n` etc).
    pub input_shape: Vec<i64>,
    /// Reshaped input shape that matmul consumes: `[1, K]`.
    pub matmul_in_shape: Vec<i64>,
    /// Weight shape as stored (may be `[K, N]` for `kn` axes or `[O, K]` for
    /// `OI` axes). `transpose_y` is set accordingly so the matmul resolves
    /// the K-dim correctly.
    pub weight_shape: Vec<i64>,
    /// Weight tensor (const, F16).
    pub weight: Tensor,
    /// True if `mb.matmul` should set `transpose_y=true` — i.e. weight is
    /// stored as `[N, K]` and we want `x @ weight.T`.
    pub transpose_weight: bool,
    /// matmul's intermediate output shape `[1, N]`.
    pub matmul_out_shape: Vec<i64>,
    /// Final output shape (matches the tract output_fact rank).
    pub output_shape: Vec<i64>,
    /// If set, the input is reduced (sum) over these axes BEFORE the matmul
    /// reshape. Used for `IHW,OI->O` patterns (reduce over H,W first,
    /// then matmul over channels).
    pub pre_reduce_axes: Option<Vec<i32>>,
    /// If set, the matmul output is tiled to `output_shape` after a
    /// `[1, N] → output_shape` reshape. Used for `I,OI->OHW` patterns
    /// (matmul produces a vector that gets broadcast across spatial dims).
    pub post_tile: bool,
    pub output_fact: TypedFact,
}

pub fn analyse_matmul(model: &TypedModel, node: &TypedNode) -> Result<MatMulAnalysis> {
    let Some(es) = node.op_as::<EinSum>() else {
        return Ok(MatMulAnalysis::Skip("not an EinSum".into()));
    };
    if es.q_params.is_some() {
        return Ok(MatMulAnalysis::Skip("quantized EinSum".into()));
    }
    if es.operating_dt != DatumType::F16 {
        return Ok(MatMulAnalysis::Skip(format!(
            "EinSum operating_dt {:?} (need F16)",
            es.operating_dt
        )));
    }
    if node.inputs.len() != 2 {
        return Ok(MatMulAnalysis::Skip(format!(
            "EinSum has {} inputs (need 2)",
            node.inputs.len()
        )));
    }

    let axes_str = format!("{}", es.axes);
    // Four matmul-shaped patterns covered:
    //   * `mkba,kn->n`   : input [M=1,K,B=1,A=1], output [N]
    //   * `k,kn->mnab`   : input [K], output [M=1,N,A=1,B=1]
    //   * `IHW,OI->O`    : input [C,H,W], output [O] — reduces H,W first
    //                      then matmul (SE-block squeeze)
    //   * `I,OI->OHW`    : input [C], output [O,H,W] — matmul then tile
    //                      to spatial (SE-block excite)
    let pattern = match axes_str.as_str() {
        "mkba,kn->n" => MatMulPattern::Rank4InRank1Out,
        "k,kn->mnab" => MatMulPattern::Rank1InRank4Out,
        "IHW,OI->O" => MatMulPattern::SpatialReduceMatmul,
        "I,OI->OHW" => MatMulPattern::MatmulSpatialBroadcast,
        other => {
            return Ok(MatMulAnalysis::Skip(format!(
                "EinSum axes {other:?} not a matmul pattern (matmul covers \
                 mkba,kn->n, k,kn->mnab, IHW,OI->O, I,OI->OHW)"
            )));
        }
    };

    let data_fact = model.outlet_fact(node.inputs[0])?;
    if data_fact.datum_type != DatumType::F16 {
        return Ok(MatMulAnalysis::Skip(format!(
            "MatMul data dtype {:?} (need F16)",
            data_fact.datum_type
        )));
    }
    let raw_in = match super::shape_to_concrete_i64(&data_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(MatMulAnalysis::Skip(format!(
                "MatMul data symbolic shape: {:?}",
                data_fact.shape
            )));
        }
    };

    // Validate shape vs pattern.
    let (k_dim, in_singletons_ok) = match pattern {
        MatMulPattern::Rank4InRank1Out => {
            if raw_in.len() != 4 {
                return Ok(MatMulAnalysis::Skip(format!(
                    "MatMul mkba,kn->n input rank {} (need 4)",
                    raw_in.len()
                )));
            }
            let ok = raw_in[0] == 1 && raw_in[2] == 1 && raw_in[3] == 1;
            (raw_in[1], ok)
        }
        MatMulPattern::Rank1InRank4Out => {
            if raw_in.len() != 1 {
                return Ok(MatMulAnalysis::Skip(format!(
                    "MatMul k,kn->mnab input rank {} (need 1)",
                    raw_in.len()
                )));
            }
            (raw_in[0], true)
        }
        MatMulPattern::SpatialReduceMatmul => {
            // IHW,OI->O: tract fact is rank 3 [C, H, W] (CHW). However the
            // MLPackage value flowing in from upstream is rank 4 [1, C, H, W]
            // because Conv with CHW data prepends N=1 in the MLPackage shape.
            // Tract's `raw_in` here is rank 3 — we use it for the K-dim
            // (= C) but the emit code reduces over MLPackage axes [2, 3].
            if raw_in.len() != 3 {
                return Ok(MatMulAnalysis::Skip(format!(
                    "MatMul IHW,OI->O input rank {} (need 3 in tract; rank 4 in MLPackage)",
                    raw_in.len()
                )));
            }
            (raw_in[0], true)
        }
        MatMulPattern::MatmulSpatialBroadcast => {
            // I,OI->OHW: input [C], output [O, H, W] — H, W from output_fact.
            if raw_in.len() != 1 {
                return Ok(MatMulAnalysis::Skip(format!(
                    "MatMul I,OI->OHW input rank {} (need 1)",
                    raw_in.len()
                )));
            }
            (raw_in[0], true)
        }
    };
    if !in_singletons_ok {
        return Ok(MatMulAnalysis::Skip(format!(
            "MatMul {axes_str:?} input shape {raw_in:?} has non-1 singleton dim — \
             generic matmul not yet implemented"
        )));
    }

    let weight = match const_tensor(model, node.inputs[1])? {
        Some(t) => t.into_owned(),
        None => {
            let n = &model.nodes[node.inputs[1].node];
            return Ok(MatMulAnalysis::Skip(format!(
                "MatMul weight (slot 1) is not Const — actual op: {}",
                n.op.name()
            )));
        }
    };
    if weight.datum_type() != DatumType::F16 {
        return Ok(MatMulAnalysis::Skip(format!(
            "MatMul weight dtype {:?} (need F16)",
            weight.datum_type()
        )));
    }
    if weight.rank() != 2 {
        return Ok(MatMulAnalysis::Skip(format!(
            "MatMul weight rank {} (need 2 for `kn` operand)",
            weight.rank()
        )));
    }
    // Weight axis layout depends on the einsum pattern. `kn` operand =>
    // weight is [K, N] (no transpose). `OI` operand => weight is [O, K]
    // (= [N, K]) and we need to transpose for the matmul.
    let weight_is_oi = matches!(
        pattern,
        MatMulPattern::SpatialReduceMatmul | MatMulPattern::MatmulSpatialBroadcast
    );
    let (k_w, n) = if weight_is_oi {
        // weight is [O, I] = [N, K]
        (weight.shape()[1] as i64, weight.shape()[0] as i64)
    } else {
        // weight is [K, N]
        (weight.shape()[0] as i64, weight.shape()[1] as i64)
    };
    if k_w != k_dim {
        return Ok(MatMulAnalysis::Skip(format!(
            "MatMul K dim mismatch: weight {k_w}, input {k_dim}"
        )));
    }

    let matmul_in_shape = vec![1, k_dim];
    let matmul_out_shape = vec![1, n];
    let (output_shape, pre_reduce_axes, post_tile) = match pattern {
        MatMulPattern::Rank4InRank1Out => (vec![n], None, false),
        MatMulPattern::Rank1InRank4Out => (vec![1, n, 1, 1], None, false),
        MatMulPattern::SpatialReduceMatmul => {
            // MLPackage upstream is rank 4 [1, C, H, W] — reduce H,W axes
            // [2, 3] (NOT tract's rank-3 axes [1, 2]). Output [O].
            (vec![n], Some(vec![2, 3]), false)
        }
        MatMulPattern::MatmulSpatialBroadcast => {
            // Get H,W from output_fact (rank 3 [O,H,W]).
            let raw_out_shape = match super::shape_to_concrete_i64(&node.outputs[0].fact.shape) {
                Some(s) => s,
                None => {
                    return Ok(MatMulAnalysis::Skip(format!(
                        "MatMul I,OI->OHW output symbolic shape: {:?}",
                        node.outputs[0].fact.shape
                    )));
                }
            };
            (raw_out_shape, None, true)
        }
    };

    let weight_shape: Vec<i64> = if weight_is_oi { vec![n, k_dim] } else { vec![k_dim, n] };

    // For SpatialReduceMatmul, the MLPackage input is rank 4 [1, C, H, W]
    // (tract surfaces it as rank-3 CHW but Conv prepended N=1 in the
    // MLPackage). Store the MLPackage-side shape so emit + reduce axes
    // line up with the actual flowing tensor.
    let plan_input_shape: Vec<i64> = match pattern {
        MatMulPattern::SpatialReduceMatmul => {
            // raw_in = [C, H, W] → MLPackage [1, C, H, W]
            std::iter::once(1).chain(raw_in.iter().copied()).collect()
        }
        _ => raw_in.clone(),
    };

    Ok(MatMulAnalysis::Translatable(MatMulPlan {
        input_shape: plan_input_shape,
        matmul_in_shape,
        weight_shape,
        weight,
        transpose_weight: weight_is_oi,
        matmul_out_shape,
        output_shape,
        pre_reduce_axes,
        post_tile,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

#[derive(Clone, Copy)]
enum MatMulPattern {
    Rank4InRank1Out,
    Rank1InRank4Out,
    SpatialReduceMatmul,
    MatmulSpatialBroadcast,
}

pub fn emit_matmul_mil(
    plan: &MatMulPlan,
    blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;
    let mut ops: Vec<mil::Operation> = Vec::new();

    // 0. Optional pre-reduce: for SpatialReduceMatmul (`IHW,OI->O`) we
    //    reduce H,W before matmul. tract's CHW input is rank 3; reduce
    //    axes [1, 2] keeps_dims=false → produces [C].
    let matmul_input_name = if let Some(axes) = &plan.pre_reduce_axes {
        let reduce_n = format!("{p}_pre_reduce");
        let axes_n = format!("{reduce_n}_axes");
        let keep_n = format!("{reduce_n}_keep");
        let i32t_axes = tensor_type(DataType::Int32, &[axes.len() as i64]);
        ops.push(op_const_immediate(&axes_n, i32t_axes, tv_ints(axes.clone())));
        ops.push(op_const_immediate(
            &keep_n,
            tensor_type_scalar(DataType::Bool),
            mil::TensorValue {
                value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                    values: vec![false],
                })),
            },
        ));
        // Compute reduced shape: input_shape with the given axes removed.
        let reduced_shape: Vec<i64> = plan
            .input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(&(*i as i32)))
            .map(|(_, &v)| v)
            .collect();
        let reduce_op = mil::Operation {
            r#type: "reduce_sum".into(),
            inputs: HashMap::from([
                ("x".into(), arg_name(input_name)),
                ("axes".into(), arg_name(&axes_n)),
                ("keep_dims".into(), arg_name(&keep_n)),
            ]),
            outputs: vec![mil::NamedValueType {
                name: reduce_n.clone(),
                r#type: Some(tensor_type(DataType::Float16, &reduced_shape)),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        };
        ops.push(reduce_op);
        reduce_n
    } else {
        input_name.to_string()
    };

    // 1. Reshape input to `[1, K]`.
    let reshape_in_n = format!("{p}_in_reshape");
    let reshape_in_shape_const = format!("{p}_in_reshape_shape");
    let i32t_2 = tensor_type(DataType::Int32, &[2]);
    let reshape_in_shape_op = op_const_immediate(
        &reshape_in_shape_const,
        i32t_2.clone(),
        tv_ints(plan.matmul_in_shape.iter().map(|&v| v as i32).collect()),
    );
    let reshape_in_op = mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&matmul_input_name)),
            ("shape".into(), arg_name(&reshape_in_shape_const)),
        ]),
        outputs: vec![mil::NamedValueType {
            name: reshape_in_n.clone(),
            r#type: Some(tensor_type(DataType::Float16, &plan.matmul_in_shape)),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    // 2. Const weight blob.
    let weight_off = blob.add(BlobDataType::Float16, plan.weight.as_bytes());
    let weight_n = format!("{p}_weight");
    let weight_ty = tensor_type(DataType::Float16, &plan.weight_shape);
    let weight_op = op_const_blob(&weight_n, weight_ty, mlpackage::WEIGHT_BLOB_PATH, weight_off);

    // 3. transpose_x / transpose_y bools.
    let bool_t = tensor_type_scalar(DataType::Bool);
    let tx_n = format!("{p}_tx");
    let ty_n = format!("{p}_ty");
    let bool_val = |v: bool| mil::TensorValue {
        value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
            values: vec![v],
        })),
    };
    let tx_op = op_const_immediate(&tx_n, bool_t.clone(), bool_val(false));
    // For `OI` weights (stored as [N, K]) transpose_y=true so `mb.matmul`
    // computes `x @ weight.T` = standard `x @ weight` with [K, N] semantics.
    let ty_op = op_const_immediate(&ty_n, bool_t, bool_val(plan.transpose_weight));

    // 4. matmul: [1, K] @ [K, N] → [1, N].
    let mm_out_n = format!("{p}_mm");
    let matmul_op = mil::Operation {
        r#type: "matmul".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&reshape_in_n)),
            ("y".into(), arg_name(&weight_n)),
            ("transpose_x".into(), arg_name(&tx_n)),
            ("transpose_y".into(), arg_name(&ty_n)),
        ]),
        outputs: vec![mil::NamedValueType {
            name: mm_out_n.clone(),
            r#type: Some(tensor_type(DataType::Float16, &plan.matmul_out_shape)),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    // 5. Reshape final output to whatever the tract output_fact expects.
    //    For the `post_tile` (MatmulSpatialBroadcast) case, reshape goes to
    //    [O, 1, 1] first; then a final tile expands to [O, H, W].
    let reshape_target_shape: Vec<i64> = if plan.post_tile {
        // Reshape [1, N] → [N, 1, 1], then tile to [O, H, W].
        vec![plan.output_shape[0], 1, 1]
    } else {
        plan.output_shape.clone()
    };
    let reshape_out_shape_const = format!("{p}_out_reshape_shape");
    let i32t_out = tensor_type(DataType::Int32, &[reshape_target_shape.len() as i64]);
    let reshape_out_shape_op = op_const_immediate(
        &reshape_out_shape_const,
        i32t_out,
        tv_ints(reshape_target_shape.iter().map(|&v| v as i32).collect()),
    );
    let reshape_out_name =
        if plan.post_tile { format!("{p}_pre_tile") } else { output_name.to_string() };
    let reshape_out_op = mil::Operation {
        r#type: "reshape".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&mm_out_n)),
            ("shape".into(), arg_name(&reshape_out_shape_const)),
        ]),
        outputs: vec![mil::NamedValueType {
            name: reshape_out_name.clone(),
            r#type: Some(tensor_type(DataType::Float16, &reshape_target_shape)),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    // Assemble.
    ops.push(reshape_in_shape_op);
    ops.push(reshape_in_op);
    ops.push(weight_op);
    ops.push(tx_op);
    ops.push(ty_op);
    ops.push(matmul_op);
    ops.push(reshape_out_shape_op);
    ops.push(reshape_out_op);

    // 6. Optional post-tile: [O, 1, 1] → [O, H, W] for MatmulSpatialBroadcast.
    if plan.post_tile {
        let reps: Vec<i32> = vec![1, plan.output_shape[1] as i32, plan.output_shape[2] as i32];
        let reps_n = format!("{p}_tile_reps");
        let reps_op =
            op_const_immediate(&reps_n, tensor_type(DataType::Int32, &[3]), tv_ints(reps));
        let tile_op = mil::Operation {
            r#type: "tile".into(),
            inputs: HashMap::from([
                ("x".into(), arg_name(&reshape_out_name)),
                ("reps".into(), arg_name(&reps_n)),
            ]),
            outputs: vec![mil::NamedValueType {
                name: output_name.to_string(),
                r#type: Some(tensor_type(DataType::Float16, &plan.output_shape)),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        };
        ops.push(reps_op);
        ops.push(tile_op);
    }

    Ok(ops)
}

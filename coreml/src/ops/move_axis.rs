//! tract `AxisOp::Move(from, to)` → MIL `transpose` translator.
//!
//! `Move(from, to)` moves a single axis from position `from` to position
//! `to`, shifting the intermediate axes by one to fill the gap. Equivalent
//! to a transpose with a permutation that's the identity except for the
//! `from`→`to` swap.
//!
//! Critical for transformer models — every windowed-attention block has a
//! Move/Transpose to bring the head dim, the window dim, and/or the
//! sequence dim into matmul-canonical position. Without this translator
//! every Move is a CPU residual that fragments the subgraph.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::change_axes::AxisOp;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum MoveAxisAnalysis {
    Translatable(MoveAxisPlan),
    Skip(String),
}

pub struct MoveAxisPlan {
    pub input_shape: Vec<i64>,
    pub output_shape: Vec<i64>,
    /// **MLPackage-boundary input shape**: `input_shape` with leading unit
    /// dims stripped if rank > 5. CoremlOp does the squeeze at the
    /// tract-side ↔ MLPackage-side boundary. Equal to `input_shape` when
    /// rank ≤ 5.
    pub input_external_shape: Vec<i64>,
    /// MLPackage-boundary output shape (same convention).
    pub output_external_shape: Vec<i64>,
    /// Permutation vector at EXTERNAL ranks: `perm[i]` = external input
    /// axis at external output position i.
    pub perm: Vec<i32>,
    pub output_fact: TypedFact,
}

pub fn analyse_move_axis(model: &TypedModel, node: &TypedNode) -> Result<MoveAxisAnalysis> {
    let Some(ax) = node.op_as::<AxisOp>() else {
        return Ok(MoveAxisAnalysis::Skip("not an AxisOp".into()));
    };
    let (from, to) = match ax {
        AxisOp::Move(f, t) => (*f, *t),
        AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Reshape(_, _, _) => {
            return Ok(MoveAxisAnalysis::Skip(
                "AxisOp::Add/Rm/Reshape handled by their dedicated translators".into(),
            ));
        }
    };

    if node.inputs.len() != 1 {
        return Ok(MoveAxisAnalysis::Skip(format!(
            "MoveAxis node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(MoveAxisAnalysis::Skip(format!(
            "MoveAxis input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(MoveAxisAnalysis::Skip(format!(
                "MoveAxis input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if input_shape.is_empty() {
        return Ok(MoveAxisAnalysis::Skip("MoveAxis input is rank 0".into()));
    }
    if from >= input_shape.len() || to >= input_shape.len() {
        return Ok(MoveAxisAnalysis::Skip(format!(
            "MoveAxis from={from} or to={to} out of range for rank {}",
            input_shape.len()
        )));
    }

    // Build the tract-rank perm vector. Identity, remove `from`, insert at `to`.
    let n_tract = input_shape.len();
    let mut tract_perm: Vec<usize> = (0..n_tract).collect();
    let elem = tract_perm.remove(from);
    tract_perm.insert(to, elem);

    // Tract output_shape by applying the same permutation.
    let output_shape: Vec<i64> = tract_perm.iter().map(|&i| input_shape[i]).collect();

    // Boundary-strip rank > 5 inputs/outputs by removing LEADING unit dims.
    let input_external_shape = match super::rank::try_strip_to_rank5(&input_shape) {
        Some(s) => s,
        None => {
            return Ok(MoveAxisAnalysis::Skip(format!(
                "MoveAxis input rank {} > 5 with no leading unit dims to strip",
                input_shape.len()
            )));
        }
    };
    let output_external_shape = match super::rank::try_strip_to_rank5(&output_shape) {
        Some(s) => s,
        None => {
            return Ok(MoveAxisAnalysis::Skip(format!(
                "MoveAxis output rank {} > 5 with no leading unit dims to strip",
                output_shape.len()
            )));
        }
    };
    let pad_in = input_shape.len() - input_external_shape.len();
    let pad_out = output_shape.len() - output_external_shape.len();
    if from < pad_in || to < pad_out {
        return Ok(MoveAxisAnalysis::Skip(format!(
            "MoveAxis(from={from}, to={to}) operates on a stripped leading unit dim; \
             can't strip safely (pad_in={pad_in}, pad_out={pad_out})",
        )));
    }
    // External perm: shift positions by the strip amount.
    let from_ext = from - pad_in;
    let to_ext = to - pad_out;
    let n_ext = input_external_shape.len();
    let mut perm: Vec<usize> = (0..n_ext).collect();
    let elem_ext = perm.remove(from_ext);
    perm.insert(to_ext, elem_ext);

    Ok(MoveAxisAnalysis::Translatable(MoveAxisPlan {
        input_shape,
        output_shape,
        input_external_shape,
        output_external_shape,
        perm: perm.iter().map(|&v| v as i32).collect(),
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_move_axis_mil(
    plan: &MoveAxisPlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    // Use EXTERNAL shapes (post-strip if rank > 5). MIL transpose runs on
    // the rank-5 (or less) external view; CoremlOp at the boundary handles
    // the squeeze/unsqueeze.
    //
    // **Defensive reshape**: only emit when our `input_external_shape` is
    // rank 5 (post-strip from rank 6+). In that case upstream's actual MIL
    // declared rank may be 4 (if upstream rank-pads to 4) or 5 (matches
    // ours). The reshape converts to our expected rank — safe in either
    // case (MIL elides no-op reshapes).
    //
    // For input_external_shape rank ≤ 4, the in-MLPackage convention is
    // rank-4-padded everywhere; both upstream and ours emit rank 4, so the
    // defensive reshape is unnecessary cost. Skipping it saves ~12-15 ms
    // per Move on hot transformer chains (19 of them in SAM 2).
    let needs_defensive_reshape = plan.input_external_shape.len() > 4;
    let mut ops: Vec<mil::Operation> = Vec::new();
    let transpose_input_name = if needs_defensive_reshape {
        let in_canon_n = format!("{output_name}_in_canon");
        let in_shape_n = format!("{in_canon_n}_shape");
        let i32t_in = tensor_type(DataType::Int32, &[plan.input_external_shape.len() as i64]);
        ops.push(op_const_immediate(
            &in_shape_n,
            i32t_in,
            tv_ints(plan.input_external_shape.iter().map(|&v| v as i32).collect()),
        ));
        ops.push(mil::Operation {
            r#type: "reshape".into(),
            inputs: HashMap::from([
                ("x".into(), arg_name(input_name)),
                ("shape".into(), arg_name(&in_shape_n)),
            ]),
            outputs: vec![mil::NamedValueType {
                name: in_canon_n.clone(),
                r#type: Some(tensor_type(DataType::Float16, &plan.input_external_shape)),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        });
        in_canon_n
    } else {
        input_name.to_string()
    };

    let perm_n = format!("{output_name}_perm");
    let i32t = tensor_type(DataType::Int32, &[plan.perm.len() as i64]);
    ops.push(op_const_immediate(&perm_n, i32t, tv_ints(plan.perm.clone())));

    let out_ty = tensor_type(DataType::Float16, &plan.output_external_shape);
    ops.push(mil::Operation {
        r#type: "transpose".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&transpose_input_name)),
            ("perm".into(), arg_name(&perm_n)),
        ]),
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    Ok(ops)
}

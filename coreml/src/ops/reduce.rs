//! tract `Reduce<Sum|Mean(via Prod-of-1)|Max|Min>` → MIL `reduce_*` translator.
//!
//! tract's `Reduce` keeps reduced axes (sets dim to 1) — matches MIL
//! `reduce_*(keep_dims=true)`. SqueezeNet has a `Reduce<Sum>` over the
//! H×W axes that's part of its global-average-pool emulation; MobileNet
//! similarly has reductions in its classifier head when the `f32_to_f16`
//! transform doesn't fold the AveragePool out.
//!
//! Phase 3 first cut: F16, reducer ∈ {Sum, Max, Min, Prod}; reducer ==
//! `MeanOfSquares` deferred (rare, needs a square+sum lower); ArgMax/ArgMin
//! deferred (output dtype change).

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::nn::{Reduce, Reducer};

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum ReduceAnalysis {
    Translatable(ReducePlan),
    Skip(String),
}

pub struct ReducePlan {
    /// MLPackage-side input shape (rank as-is — no padding; Reduce preserves rank).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (same rank, reduced axes set to 1).
    pub output_shape: Vec<i64>,
    /// MIL op name: "reduce_sum", "reduce_max", etc.
    pub mil_op: &'static str,
    /// Axes (in tract / MLPackage rank coords).
    pub axes: Vec<i32>,
    /// tract-side output fact.
    pub output_fact: TypedFact,
    /// True for `Reducer::MeanOfSquares` — emit as `square(x)` then
    /// `reduce_mean`, since MIL has no direct `mean_of_squares` op.
    pub mean_of_squares: bool,
}

pub fn analyse_reduce(model: &TypedModel, node: &TypedNode) -> Result<ReduceAnalysis> {
    let Some(red) = node.op_as::<Reduce>() else {
        return Ok(ReduceAnalysis::Skip("not a Reduce".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(ReduceAnalysis::Skip(format!(
            "Reduce node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let (mil_op, mean_of_squares) = match red.reducer {
        Reducer::Sum => ("reduce_sum", false),
        Reducer::Max => ("reduce_max", false),
        Reducer::Min => ("reduce_min", false),
        Reducer::Prod => ("reduce_prod", false),
        // No direct MIL op — emit `square(x)` then `reduce_mean`.
        Reducer::MeanOfSquares => ("reduce_mean", true),
        other => {
            return Ok(ReduceAnalysis::Skip(format!(
                "Reduce<{other:?}> not supported (Phase 3 covers Sum/Max/Min/Prod/MeanOfSquares)"
            )));
        }
    };

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(ReduceAnalysis::Skip(format!(
            "Reduce input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let raw_in = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(ReduceAnalysis::Skip(format!(
                "Reduce input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if !(2..=5).contains(&raw_in.len()) {
        return Ok(ReduceAnalysis::Skip(format!(
            "Reduce input rank {} (only 2..=5 supported)",
            raw_in.len()
        )));
    }

    // **Rank-padding correction.** Most upstream translators (Conv, BinOp,
    // MaxPool, AvgPool, EinSum-as-conv, Resize) normalise their MLPackage
    // outputs to rank 4 by prepending N=1 if the tract fact is rank 3 (CHW).
    // tract-side reduce axes therefore need to be shifted by `(4 - tract_rank)`
    // to match the MLPackage tensor that actually flows in. Without this,
    // reducing axes [1, 2] of a tract rank-3 [C, H, W] becomes reducing [1, 2]
    // of MLPackage rank-4 [1, C, H, W] which collapses C instead of H/W.
    let pad = 4_usize.saturating_sub(raw_in.len());
    let input_shape = super::rank::pad_to_rank_4(&raw_in);

    let mut axes: Vec<i32> = red.axes.iter().map(|&a| a as i32 + pad as i32).collect();
    axes.sort_unstable();
    if axes.iter().any(|&a| a < 0 || (a as usize) >= input_shape.len()) {
        return Ok(ReduceAnalysis::Skip(format!(
            "Reduce axes {:?} out of range for rank {}",
            axes,
            input_shape.len()
        )));
    }

    // Output: same rank, reduced axes set to 1.
    let mut output_shape = input_shape.clone();
    for &a in &axes {
        output_shape[a as usize] = 1;
    }

    Ok(ReduceAnalysis::Translatable(ReducePlan {
        input_shape,
        output_shape,
        mil_op,
        axes,
        output_fact: node.outputs[0].fact.clone(),
        mean_of_squares,
    }))
}

pub fn emit_reduce_mil(
    plan: &ReducePlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let i32t_axes = tensor_type(DataType::Int32, &[plan.axes.len() as i64]);
    let bool_t = tensor_type(DataType::Bool, &[]);

    let axes_n = format!("{output_name}_axes");
    let keep_n = format!("{output_name}_keep_dims");

    let axes_op = op_const_immediate(&axes_n, i32t_axes, tv_ints(plan.axes.clone()));
    let keep_op = op_const_immediate(
        &keep_n,
        bool_t,
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: vec![true],
            })),
        },
    );

    // For MeanOfSquares, emit `square(x)` first; the reduce_mean consumes the
    // squared intermediate. Output type of square = input type/shape.
    let mut prelude = vec![axes_op, keep_op];
    let reduce_input_name = if plan.mean_of_squares {
        let sq_name = format!("{output_name}_squared");
        let mut sq_inputs: HashMap<String, mil::Argument> = HashMap::new();
        sq_inputs.insert("x".into(), arg_name(input_name));
        let sq_ty = tensor_type(DataType::Float16, &plan.input_shape);
        let sq_op = mil::Operation {
            r#type: "square".into(),
            inputs: sq_inputs,
            outputs: vec![mil::NamedValueType { name: sq_name.clone(), r#type: Some(sq_ty) }],
            blocks: vec![],
            attributes: HashMap::new(),
        };
        prelude.push(sq_op);
        sq_name
    } else {
        input_name.to_string()
    };

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(&reduce_input_name));
    inputs.insert("axes".into(), arg_name(&axes_n));
    inputs.insert("keep_dims".into(), arg_name(&keep_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let red_op = mil::Operation {
        r#type: plan.mil_op.into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    prelude.push(red_op);
    Ok(prelude)
}

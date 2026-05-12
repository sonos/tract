//! tract `Slice` → MIL `slice_by_index` translator.
//!
//! tract's `Slice` slices a single axis with `start..end` (no stride, no
//! negative indices, no multi-axis). MIL's `slice_by_index` is more general
//! (multi-axis with per-axis begin/end/stride and bitmasks), so we map by:
//!   begin[i] = (i == axis) ? start : 0; begin_mask bit i = (i != axis)
//!   end[i]   = (i == axis) ? end   : 0; end_mask   bit i = (i != axis)
//!   stride[i] = 1; squeeze_mask = 0 (preserve rank)
//!
//! Phase 3: F16 only, rank 2..=5, concrete `start`/`end` (symbolic dims
//! deferred — we'd need to evaluate the TDim against bound symbols).

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::array::Slice;

use crate::mil::blob::BlobBuilder;
use crate::mil::op::{arg_name, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tv_ints};
use crate::proto::core_ml::specification::mil_spec as mil;

use super::shape_to_concrete_i64;

#[allow(clippy::large_enum_variant)]
pub enum SliceAnalysis {
    Translatable(SlicePlan),
    Skip(String),
}

pub struct SlicePlan {
    /// MLPackage-side input shape (rank N — natural rank, no padding).
    pub input_shape: Vec<i64>,
    /// MLPackage-side output shape (rank N; `axis` dim becomes `end - start`).
    pub output_shape: Vec<i64>,
    /// Axis being sliced (in input rank coords).
    pub axis: usize,
    pub start: i32,
    pub end: i32,
    pub output_fact: TypedFact,
}

pub fn analyse_slice(model: &TypedModel, node: &TypedNode) -> Result<SliceAnalysis> {
    let Some(sl) = node.op_as::<Slice>() else {
        return Ok(SliceAnalysis::Skip("not a Slice".into()));
    };
    if node.inputs.len() != 1 {
        return Ok(SliceAnalysis::Skip(format!(
            "Slice node has {} inputs (need 1)",
            node.inputs.len()
        )));
    }

    let in_fact = model.outlet_fact(node.inputs[0])?;
    if in_fact.datum_type != DatumType::F16 {
        return Ok(SliceAnalysis::Skip(format!(
            "Slice input dtype {:?} (need F16)",
            in_fact.datum_type
        )));
    }
    let input_shape = match shape_to_concrete_i64(&in_fact.shape) {
        Some(s) => s,
        None => {
            return Ok(SliceAnalysis::Skip(format!(
                "Slice input symbolic shape: {:?}",
                in_fact.shape
            )));
        }
    };
    if !(2..=5).contains(&input_shape.len()) {
        return Ok(SliceAnalysis::Skip(format!(
            "Slice input rank {} (only 2..=5 supported)",
            input_shape.len()
        )));
    }
    if sl.axis >= input_shape.len() {
        return Ok(SliceAnalysis::Skip(format!(
            "Slice axis {} >= input rank {}",
            sl.axis,
            input_shape.len()
        )));
    }

    // start / end must resolve to concrete i64.
    let start_i64 = match sl.start.to_i64() {
        Ok(v) => v,
        Err(_) => {
            return Ok(SliceAnalysis::Skip(format!("Slice start is symbolic: {:?}", sl.start)));
        }
    };
    let end_i64 = match sl.end.to_i64() {
        Ok(v) => v,
        Err(_) => {
            return Ok(SliceAnalysis::Skip(format!("Slice end is symbolic: {:?}", sl.end)));
        }
    };
    if start_i64 < 0 || end_i64 < start_i64 || end_i64 > input_shape[sl.axis] {
        return Ok(SliceAnalysis::Skip(format!(
            "Slice {start_i64}..{end_i64} out of bounds for axis {} (dim {})",
            sl.axis, input_shape[sl.axis]
        )));
    }

    let mut output_shape = input_shape.clone();
    output_shape[sl.axis] = end_i64 - start_i64;

    Ok(SliceAnalysis::Translatable(SlicePlan {
        input_shape,
        output_shape,
        axis: sl.axis,
        start: start_i64 as i32,
        end: end_i64 as i32,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

pub fn emit_slice_mil(
    plan: &SlicePlan,
    _blob: &mut BlobBuilder,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let rank = plan.input_shape.len();
    let i32t_rank = tensor_type(DataType::Int32, &[rank as i64]);
    let bool_t_rank = tensor_type(DataType::Bool, &[rank as i64]);

    // begin / end / stride vectors (rank-length).
    let mut begin: Vec<i32> = vec![0; rank];
    let mut end: Vec<i32> = vec![0; rank];
    begin[plan.axis] = plan.start;
    end[plan.axis] = plan.end;
    let stride: Vec<i32> = vec![1; rank];

    // begin_mask / end_mask / squeeze_mask: per-axis bool vectors (NOT int32
    // bitmasks). begin_mask[i]=true ⇒ ignore begin[i] and use full range.
    // squeeze_mask[i]=true ⇒ drop axis i after slicing (we always preserve
    // rank, so all false).
    let begin_mask: Vec<bool> = (0..rank).map(|i| i != plan.axis).collect();
    let end_mask: Vec<bool> = (0..rank).map(|i| i != plan.axis).collect();
    let squeeze_mask: Vec<bool> = vec![false; rank];

    let begin_n = format!("{output_name}_begin");
    let end_n = format!("{output_name}_end");
    let stride_n = format!("{output_name}_stride");
    let bmask_n = format!("{output_name}_begin_mask");
    let emask_n = format!("{output_name}_end_mask");
    let smask_n = format!("{output_name}_squeeze_mask");

    let begin_op = op_const_immediate(&begin_n, i32t_rank.clone(), tv_ints(begin));
    let end_op = op_const_immediate(&end_n, i32t_rank.clone(), tv_ints(end));
    let stride_op = op_const_immediate(&stride_n, i32t_rank, tv_ints(stride));
    let bmask_op = op_const_immediate(
        &bmask_n,
        bool_t_rank.clone(),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: begin_mask,
            })),
        },
    );
    let emask_op = op_const_immediate(
        &emask_n,
        bool_t_rank.clone(),
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: end_mask,
            })),
        },
    );
    let smask_op = op_const_immediate(
        &smask_n,
        bool_t_rank,
        mil::TensorValue {
            value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
                values: squeeze_mask,
            })),
        },
    );

    // Defensive input-rank normalization. analyse_slice uses tract's literal
    // input rank (3 for CHW, etc.) but upstream MIL ops may emit at a
    // different rank (e.g. activations pad to rank-4, Conv emits rank-4
    // output). MIL `slice_by_index` requires `begin`/`end`/`stride` lengths
    // to match the input rank; without aligning we get
    // "Dimension 0 of tensor parameter begin[0] has unexpected length N;
    //  expected M". YOLOv8 surfaced this — slice consumes a rank-4 softmax
    // output but its begin/end are rank-3 (tract-side).
    let in_rank4_name = format!("{output_name}_x_rank{rank}");
    let in_shape_n = format!("{output_name}_x_shape");
    let in_shape_op = op_const_immediate(
        &in_shape_n,
        tensor_type(DataType::Int32, &[plan.input_shape.len() as i64]),
        tv_ints(plan.input_shape.iter().map(|&v| v as i32).collect()),
    );
    let in_ty = tensor_type(DataType::Float16, &plan.input_shape);
    let mut reshape_inputs: HashMap<String, mil::Argument> = HashMap::new();
    reshape_inputs.insert("x".into(), arg_name(input_name));
    reshape_inputs.insert("shape".into(), arg_name(&in_shape_n));
    let in_reshape_op = mil::Operation {
        r#type: "reshape".into(),
        inputs: reshape_inputs,
        outputs: vec![mil::NamedValueType { name: in_rank4_name.clone(), r#type: Some(in_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    let mut inputs: HashMap<String, mil::Argument> = HashMap::new();
    inputs.insert("x".into(), arg_name(&in_rank4_name));
    inputs.insert("begin".into(), arg_name(&begin_n));
    inputs.insert("end".into(), arg_name(&end_n));
    inputs.insert("stride".into(), arg_name(&stride_n));
    inputs.insert("begin_mask".into(), arg_name(&bmask_n));
    inputs.insert("end_mask".into(), arg_name(&emask_n));
    inputs.insert("squeeze_mask".into(), arg_name(&smask_n));

    let out_ty = tensor_type(DataType::Float16, &plan.output_shape);
    let slice_op = mil::Operation {
        r#type: "slice_by_index".into(),
        inputs,
        outputs: vec![mil::NamedValueType { name: output_name.to_string(), r#type: Some(out_ty) }],
        blocks: vec![],
        attributes: HashMap::new(),
    };

    Ok(vec![
        begin_op,
        end_op,
        stride_op,
        bmask_op,
        emask_op,
        smask_op,
        in_shape_op,
        in_reshape_op,
        slice_op,
    ])
}

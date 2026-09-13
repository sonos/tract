use crate::model::{ParseResult, ParsingContext};
use crate::pb::*;
use crate::tract_core::ops::scan::ScanInfo;
use tract_hir::internal::*;

use tract_hir::ops;

pub fn scan(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let num_scan_inputs = node.get_attr("num_scan_inputs")?;
    let graph: &GraphProto = node.get_attr("body")?;
    let ParseResult { mut model, unresolved_inputs, .. } = ctx.parse_graph(graph)?;
    let scan_input_axes =
        node.get_attr_opt_vec("scan_input_axes")?.unwrap_or_else(|| vec![0; num_scan_inputs]);
    let closure_inputs = unresolved_inputs.len();
    let num_hidden_state = model.input_outlets()?.len() - closure_inputs - num_scan_inputs;
    let num_scan_outputs = model.output_outlets()?.len() - num_hidden_state;
    let scan_output_axes =
        node.get_attr_opt_vec("scan_output_axes")?.unwrap_or_else(|| vec![0; num_scan_outputs]);
    let scan_input_directions: Vec<i64> =
        node.get_attr_opt_vec("scan_input_directions")?.unwrap_or_else(|| vec![0; num_scan_inputs]);
    let scan_output_directions: Vec<i64> = node
        .get_attr_opt_vec("scan_output_directions")?
        .unwrap_or_else(|| vec![0; num_scan_outputs]);

    let mut mapped_inputs = vec![];
    let mut mapped_outputs = vec![];
    for ix in 0..num_hidden_state {
        mapped_inputs.push(ops::scan::InputMapping::State);
        mapped_outputs.push(ops::scan::OutputMapping {
            state: true,
            last_value_slot: Some(ix),
            scan: None,
            full_dim_hint: None,
        });
    }

    for (ix, ax) in scan_input_axes.iter().enumerate() {
        let op = expand(ops::array::RmDims::new(vec![*ax]));
        let outlet = model.input_outlets()?[num_hidden_state + ix];
        let slice_name = &model.node(outlet.node).name;
        let axis = scan_axis(*ax, graph.input.iter().find(|i| &i.name == slice_name))?;
        let chunk = scan_chunk(&scan_input_directions, ix);
        InferenceModelPatch::intercept(
            &model,
            outlet,
            format!("{}.input-{}.adjust-dim", node.name, ix),
            op,
            model.outlet_fact(outlet)?.clone(),
        )?
        .apply(&mut model)?;
        model.set_outlet_fact(outlet, InferenceFact::default())?;
        mapped_inputs.push(ops::scan::InputMapping::Scan(ScanInfo { axis, chunk }));
    }

    for _input in unresolved_inputs.iter() {
        mapped_inputs.push(ops::scan::InputMapping::Full);
    }

    for (ix, ax) in scan_output_axes.iter().enumerate() {
        let op = ops::array::AddDims::new(vec![*ax]);
        let outlet = model.output_outlets()?[num_hidden_state + ix];
        let axis = scan_axis(*ax, graph.output.get(num_hidden_state + ix))?;
        let chunk = scan_chunk(&scan_output_directions, ix);
        InferenceModelPatch::intercept(
            &model,
            outlet,
            format!("{}.output-{}-adjust-dim", node.name, ix),
            expand(op),
            InferenceFact::default(),
        )?
        .apply(&mut model)?;
        mapped_outputs.push(ops::scan::OutputMapping {
            state: false,
            scan: Some((ix + num_hidden_state, ScanInfo { axis, chunk })),
            full_dim_hint: None,
            last_value_slot: None,
        });
    }

    Ok((
        Box::new(ops::scan::InferenceScan::new(
            model,
            mapped_inputs,
            mapped_outputs,
            true,
            GenericFactoid::default(),
        )),
        unresolved_inputs,
    ))
}

/// Resolves an ONNX scan axis against the full scanned tensor, whose rank is one more than the
/// body slice's. Negative axes need the body graph to declare the slice's shape.
fn scan_axis(axis: isize, slice: Option<&ValueInfoProto>) -> TractResult<usize> {
    if axis >= 0 {
        return Ok(axis as usize);
    }
    let rank = slice
        .and_then(|s| s.r#type.as_ref())
        .and_then(|t| t.value.as_ref())
        .and_then(|type_proto::Value::TensorType(t)| t.shape.as_ref())
        .map(|shape| shape.dim.len() as isize + 1)
        .context("Scan: a negative scan axis needs the body graph to declare the slice shape")?;
    ensure!(axis >= -rank, "Scan: axis {axis} out of range for rank {rank}");
    Ok((axis + rank) as usize)
}

/// ONNX direction 1 scans in reverse, which tract expresses as a negative chunk.
fn scan_chunk(directions: &[i64], ix: usize) -> isize {
    if directions.get(ix) == Some(&1) { -1 } else { 1 }
}

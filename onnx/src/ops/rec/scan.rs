use crate::model::{ParseResult, ParsingContext};
use crate::pb::*;
use tract_core::internal::*;

use tract_core::ops::rec::scan::Scan;

pub fn scan(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<InferenceOp>, Vec<String>)> {
    let num_scan_inputs = node.get_attr("num_scan_inputs")?;
    let graph: &GraphProto = node.get_attr("body")?;
    let scan_input_axes = node.get_attr_opt_vec("scan_input_axes")?.unwrap_or(Vec::<usize>::new());
    let scan_output_axes =
        node.get_attr_opt_vec("scan_output_axes")?.unwrap_or(Vec::<usize>::new());
    let ParseResult { model, unresolved_inputs, .. } = ctx.parse_graph(graph)?;
    let scan_output_len_hints = vec!(None; scan_output_axes.len());
    Ok((
        Box::new(Scan::new(
            model,
            num_scan_inputs,
            unresolved_inputs.len(),
            scan_input_axes,
            scan_output_axes,
            scan_output_len_hints,
            true,
        )),
        unresolved_inputs,
    ))
}

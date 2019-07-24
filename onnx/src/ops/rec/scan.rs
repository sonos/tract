use crate::model::{ParseResult, ParsingContext};
use crate::pb::*;
use tract_core::internal::*;

use tract_core::ops::scan::Inference;

pub fn scan(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<InferenceOp>, Vec<String>)> {
    let num_scan_inputs = node.get_attr("num_scan_inputs")?;
    let graph: &GraphProto = node.get_attr("body")?;
    let ParseResult { mut model, unresolved_inputs, .. } = ctx.parse_graph(graph)?;
    let scan_input_axes =
        node.get_attr_opt_vec("scan_input_axes")?.unwrap_or(vec![0; num_scan_inputs]);
    let closure_inputs = unresolved_inputs.len();
    let num_hidden_state = model.input_outlets()?.len() - closure_inputs - num_scan_inputs;
    let num_scan_outputs = model.output_outlets()?.len() - num_hidden_state;
    let scan_output_axes =
        node.get_attr_opt_vec("scan_output_axes")?.unwrap_or(vec![0; num_scan_outputs]);
    let scan_output_len_hints = vec![None; scan_output_axes.len()];

    for input in 0..num_scan_inputs {
        let op = tract_core::ops::array::RmDims::new(vec![scan_input_axes[input]]);
        let outlet = model.input_outlets()?[num_hidden_state + input];
        InferenceModelPatch::intercept(
            &model,
            outlet,
            format!("input-{}-adjust-dim", input),
            op,
            model.outlet_fact(outlet)?.clone(),
    )?
        .apply(&mut model)?;
        model.set_outlet_fact(outlet, TensorFact::default())?;
    }

    for output in 0..num_scan_outputs {
        let op = tract_core::ops::array::AddDims::new(vec![scan_output_axes[output]]);
        InferenceModelPatch::intercept(
            &model,
            model.output_outlets()?[num_hidden_state + output],
            format!("output-{}-adjust-dim", output),
            op,
            TensorFact::default(),
        )?
        .apply(&mut model)?;
    }

    Ok((
        Box::new(Inference::new(
            model,
            unresolved_inputs.len(),
            scan_input_axes,
            scan_output_axes,
            scan_output_len_hints,
        )),
        unresolved_inputs,
    ))
}

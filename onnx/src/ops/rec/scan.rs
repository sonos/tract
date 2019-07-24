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

    let mut mapped_inputs = vec![];
    for ix in 0..num_hidden_state {
        mapped_inputs.push(tract_core::ops::scan::InputMapping::State {
            initializer: tract_core::ops::scan::StateInitializer::FromInput(ix),
        });
    }

    for (ix, ax) in scan_input_axes.iter().enumerate() {
        let op = tract_core::ops::array::RmDims::new(vec![*ax]);
        let outlet = model.input_outlets()?[num_hidden_state + ix];
        InferenceModelPatch::intercept(
            &model,
            outlet,
            format!("input-{}-adjust-dim", ix),
            op,
            model.outlet_fact(outlet)?.clone(),
        )?
        .apply(&mut model)?;
        model.set_outlet_fact(outlet, TensorFact::default())?;
        mapped_inputs.push(tract_core::ops::scan::InputMapping::Scan { axis: *ax, slot: ix + num_hidden_state, chunk: () });
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
            num_hidden_state,
            mapped_inputs,
            scan_output_axes,
            scan_output_len_hints,
        )),
        unresolved_inputs,
    ))
}

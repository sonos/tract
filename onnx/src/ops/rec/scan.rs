use crate::model::{ParseResult, ParsingContext};
use crate::pb::*;
use tract_core::internal::*;
use tract_core::infer::*;

use tract_core::ops::scan::InferenceScan;

pub fn scan(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
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

    let mut mapped_inputs = vec![];
    let mut mapped_outputs = vec![];
    for ix in 0..num_hidden_state {
        mapped_inputs.push(tract_core::ops::scan::InputMapping::State {
            initializer: tract_core::ops::scan::StateInitializer::FromInput(ix),
        });
        mapped_outputs.push(tract_core::ops::scan::OutputMapping {
            state: true,
            last_value_slot: Some(ix),
            full_slot: None,
            axis: 0,
            chunk: (),
            full_dim_hint: None,
        });
    }

    for (ix, ax) in scan_input_axes.iter().enumerate() {
        let op = tract_core::hir::array::RmDims::new(vec![*ax]);
        let outlet = model.input_outlets()?[num_hidden_state + ix];
        InferenceModelPatch::intercept(
            &model,
            outlet,
            format!("input-{}-adjust-dim", ix),
            op,
            model.outlet_fact(outlet)?.clone(),
        )?
        .apply(&mut model)?;
        model.set_outlet_fact(outlet, InferenceFact::default())?;
        mapped_inputs.push(tract_core::ops::scan::InputMapping::Scan {
            axis: *ax,
            slot: ix + num_hidden_state,
            chunk: (),
        });
    }

    for (ix, ax) in scan_output_axes.iter().enumerate() {
        let op = tract_core::hir::array::AddDims::new(vec![*ax]);
        let outlet = model.output_outlets()?[num_hidden_state + ix];
        InferenceModelPatch::intercept(
            &model,
            outlet,
            format!("output-{}-adjust-dim", ix),
            op,
            InferenceFact::default(),
        )?
        .apply(&mut model)?;
        mapped_outputs.push(tract_core::ops::scan::OutputMapping {
            state: false,
            axis: *ax,
            full_slot: Some(ix + num_hidden_state),
            chunk: (),
            full_dim_hint: None,
            last_value_slot: None,
        });
    }

    Ok((
        Box::new(InferenceScan::new(
            model,
            mapped_inputs,
            mapped_outputs,
            None,
            true,
            GenericFactoid::default(),
        )),
        unresolved_inputs,
    ))
}

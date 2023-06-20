use crate::model::{ParseResult, ParsingContext};
use crate::pb::*;
use crate::tract_core::ops::scan::ScanInfo;
use tract_hir::internal::*;

use tract_hir::ops;
use tract_hir::tract_core::ops::array::DynSlice;

pub fn scan(
    ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let num_scan_inputs = node.get_attr("num_scan_inputs")?;
    let graph: &GraphProto = node.get_attr("body")?;
    let ParseResult { model: mut body, unresolved_inputs, .. } = ctx.parse_graph(graph)?;
    let scan_input_axes =
        node.get_attr_opt_vec("scan_input_axes")?.unwrap_or_else(|| vec![0; num_scan_inputs]);
    let closure_inputs = unresolved_inputs.len();
    let num_hidden_state = body.input_outlets()?.len() - closure_inputs - num_scan_inputs;
    let num_scan_outputs = body.output_outlets()?.len() - num_hidden_state;
    let scan_output_axes =
        node.get_attr_opt_vec("scan_output_axes")?.unwrap_or_else(|| vec![0; num_scan_outputs]);

    let mut mapped_inputs = vec![];
    let mut mapped_outputs = vec![];
    let mut state_initializers = vec![];
    for ix in 0..num_hidden_state {
        mapped_inputs.push(ops::scan::InputMapping::State);
        state_initializers.push(None);
        mapped_outputs.push(ops::scan::OutputMapping {
            state: true,
            last_value_slot: Some(ix),
            scan: None,
            full_dim_hint: None,
        });
    }
    mapped_inputs.push(ops::scan::InputMapping::State);
    mapped_outputs.push(ops::scan::OutputMapping {
        state: true,
        last_value_slot: None,
        scan: None,
        full_dim_hint: None,
    });
    state_initializers.push(Some(rctensor0(0i64)));
    let scan_i = body.add_source("i", i64::scalar_fact().into())?;
    let one = body.add_const("one", rctensor0(1i64))?;
    let scan_i_plus_one =
        body.wire_node("i_plus_one", ops::math::Add.into_hir(), &[scan_i, one])?[0];

    // move "i" input right after the actual states instead of at the end
    body.inputs.pop();
    body.inputs.insert(num_hidden_state, scan_i);
    body.outputs.insert(num_hidden_state, scan_i_plus_one);

    let name = &node.name;
    for (ix, axis) in scan_input_axes.iter().map(|axis| *axis as usize).enumerate() {
        let outlet = body.input_outlets()?[num_hidden_state + 1 + ix];
        body.set_outlet_fact(outlet, InferenceFact::default())?;
        let mut patch = InferenceModelPatch::default();
        let x = patch.tap_model(&body, outlet)?;
        let scan_i = patch.tap_model(&body, scan_i)?;
        let scan_i_plus_one = patch.tap_model(&body, scan_i_plus_one)?;
        let x = patch.wire_node(
            format!("{name}.input-{ix}.slice"),
            DynSlice { axis, len: 1.to_dim() },
            &[x, scan_i, scan_i_plus_one],
        )?[0];
        let x = patch.wire_node(
            format!("{name}.input-{ix}.adjust_dim"),
            expand(ops::array::RmDims::new(vec![axis as isize])),
            &[x],
        )?[0];
        patch.shunt_outside(&body, outlet, x)?;
        patch.apply(&mut body)?;
        mapped_inputs.push(ops::scan::InputMapping::Full);
    }

    for _input in unresolved_inputs.iter() {
        mapped_inputs.push(ops::scan::InputMapping::Full);
    }

    for (ix, ax) in scan_output_axes.iter().enumerate() {
        let op = ops::array::AddDims::new(vec![*ax]);
        let outlet = body.output_outlets()?[num_hidden_state + 1 + ix];
        InferenceModelPatch::intercept(
            &body,
            outlet,
            format!("{}.output-{}-adjust-dim", node.name, ix),
            expand(op),
            InferenceFact::default(),
        )?
        .apply(&mut body)?;
        mapped_outputs.push(ops::scan::OutputMapping {
            state: false,
            scan: Some((ix + num_hidden_state, ScanInfo { axis: *ax as usize, chunk: 1 })),
            full_dim_hint: None,
            last_value_slot: None,
        });
    }
    Ok((
        Box::new(ops::scan::InferenceScan::new(
            body,
            mapped_inputs,
            state_initializers,
            mapped_outputs,
            true,
            GenericFactoid::default(),
        )),
        unresolved_inputs,
    ))
}

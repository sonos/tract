use tract_core::ndarray::{ArrayD, Axis};
use tract_itertools::Itertools;

use tract_core::model::OutletId;
use tract_core::plan::SimpleState;

use tract_pulse::internal::*;

use tract_libcli::annotations::Annotations;
use tract_libcli::display_params::DisplayParams;

use crate::dump::annotate_with_graph_def;
use crate::{Parameters, TractResult};
use tract_libcli::terminal;

pub fn handle(params: &Parameters, options: &DisplayParams) -> TractResult<()> {
    let decl = params
        .reference_model
        .as_deref()
        .context("Decluttered model not generated. (using --pass ?)")?
        .downcast_ref::<TypedModel>()
        .unwrap();
    let pulsed =
        params.pulsed_model.as_ref().context("Pulsed model not generated. (using --pass ?)")?;
    let model = params
        .tract_model
        .downcast_ref::<TypedModel>()
        .context("Final model is not Typed. (using --pass ?)")?;

    let decl_input_fact = decl.input_fact(0)?;
    let pulsed_input_fact = pulsed.input_fact(0)?;
    let input_pulse = pulsed_input_fact.pulse().unwrap().to_usize().unwrap();

    let mut annotations = Annotations::from_model(&*params.tract_model)?;
    annotate_with_graph_def(&mut annotations, &*params.tract_model, &params.graph)?;

    let eval_order = tract_core::model::order::eval_order_opt_ram(decl)?;

    for &decl_node in eval_order.iter() {
        let pulsed_node = match pulsed.node_by_name(&*decl.node(decl_node).name) {
            Ok(node) => node.id,
            _ => continue,
        };
        let node = match model.node_by_name(&*decl.node(decl_node).name) {
            Ok(node) => node.id,
            _ => continue,
        };
        for output_slot in 0..decl.node(decl_node).outputs.len() {
            debug!("checking node: {} output: {}", decl.node(decl_node).name, output_slot);
            let decl_outlet = OutletId::new(decl_node, output_slot);
            let pulsed_outlet = OutletId::new(pulsed_node, output_slot);
            let outlet = OutletId::new(node, output_slot);

            let pulsed_output_fact = pulsed.outlet_fact(pulsed_outlet)?;

            let stream = pulsed_output_fact.stream.as_ref().unwrap();
            let output_pulse = pulsed_output_fact.pulse().unwrap().to_usize()?;
            let output_axis = stream.axis;
            let delay = stream.delay;

            let stream_dim = delay + 3 * input_pulse + input_pulse / 2;
            let stream_symbol = model.symbols.sym("S");

            let fixed_input =
                tract_libcli::tensor::tensor_for_fact(decl_input_fact, Some(stream_dim), None)?;

            let decl = (*decl).clone();
            let fixed_result = decl
                .with_output_outlets(&[decl_outlet])?
                .concretize_dims(&SymbolValues::default().with(&stream_symbol, stream_dim as _))?
                .into_runnable()?
                .run(tvec!(fixed_input.clone().into_tvalue()))?
                .remove(output_slot);
            let fixed_output_len = fixed_result.shape()[output_axis];

            let plan = model.clone().with_output_outlets(&[outlet])?.into_runnable()?;
            let mut state = SimpleState::new(&plan)?;

            for i in 0.. {
                let input_shape = pulsed_input_fact
                    .shape
                    .iter()
                    .map(|d| d.to_usize())
                    .collect::<TractResult<TVec<_>>>()?;
                let mut pulsed_input = ArrayD::from_elem(&*input_shape, f32::NAN);
                let offset = i * input_pulse;
                if offset < stream_dim {
                    let count = input_pulse.min(stream_dim - offset);
                    pulsed_input.slice_axis_mut(Axis(stream.axis), (0..count).into()).assign(
                        &fixed_input
                            .to_array_view::<f32>()?
                            .slice_axis(Axis(stream.axis), (offset..offset + count).into()),
                    );
                };
                if offset + input_pulse > stream_dim {
                    debug!("Set known_stream_len: {stream_dim}");
                    state.session_state.resolved_symbols.set(&stream_symbol, stream_dim as _);
                };

                let output =
                    state.run(tvec!(pulsed_input.into_tensor().into()))?.remove(output_slot);

                let output_offset = output_pulse;
                let (f_o, p_o, count) = if output_offset + output_pulse <= delay {
                    // entire pulse before signal, wait
                    continue;
                } else if output_offset >= delay + fixed_output_len {
                    // entire pulse after signal, we stop
                    break;
                } else if output_offset < delay {
                    // beginning of signal
                    let count = output_pulse + output_offset - delay;
                    (0, output_pulse - count, count)
                } else if output_offset + output_pulse > delay + fixed_output_len {
                    // end of signal
                    let count = fixed_output_len + delay - output_offset;
                    (output_offset - delay, 0, count)
                } else {
                    (output_offset - delay, 0, output_pulse)
                };
                let valid_pulse_result = output.slice(output_axis, p_o, p_o + count)?;
                let valid_fixed_result = fixed_result.slice(output_axis, f_o, f_o + count)?;
                if valid_pulse_result != valid_fixed_result {
                    terminal::render_node(
                        &*params.tract_model,
                        pulsed_node,
                        &annotations,
                        options,
                    )?;
                    println!("pulse: {} ({}..{})", i, i * output_pulse, (i + 1) * output_pulse);
                    println!("expected: {:?}", &valid_fixed_result.as_slice::<f32>()?[0..10]);
                    println!(
                        "expected: {}",
                        valid_fixed_result
                            .to_array_view::<f32>()?
                            .axis_iter(Axis(output_axis))
                            .map(|s| *s.iter().next().unwrap())
                            .join(" ")
                    );
                    println!("got: {:?}", &valid_pulse_result.as_slice::<f32>()?[0..10]);
                    println!(
                        "got: {}",
                        valid_pulse_result
                            .to_array_view::<f32>()?
                            .axis_iter(Axis(output_axis))
                            .map(|s| *s.iter().next().unwrap())
                            .join(" ")
                    );
                    bail!("Pulse check failed")
                }
            }
        }
    }

    Ok(())
}

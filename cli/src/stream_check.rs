use colored::Colorize;
use errors::*;
use ndarray::Axis;
use simplelog::Level::Info;
use tract::analyser::TensorFact;
use tract::streaming::prelude::*;
use tract::{SimplePlan, Tensor};
use tract::plan::SimpleState;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, output_params: OutputParameters) -> CliResult<()> {
    let model = params.tfd_model;
    let input = model.input_fact()?;

    // First generate random values for the inputs.
    let fixed_input = ::tensor::make_inputs_stream(&[input.clone()], 500)?.remove(0);
    let wanted_matches = 20;

    let regular_input_fact = TensorFact::default()
        .with_shape(fixed_input.shape())
        .with_datum_type(fixed_input.datum_type());

    // streaming model
    let mut stream_model = model.clone();
    stream_model.set_fact(model.inputs()?[0], input.clone())?;
    stream_model.analyse()?;
    /*
    let stream_model = model.analyser()?
        .with_input_hint(input)?
        .to_optimized_model()?;
        */

    // batch model
    /*
    let batch_model = model.analyser()?
        .with_input_hint(&regular_input_fact)?
        .to_optimized_model()?;
    */
    let mut batch_model = model.clone();
    batch_model.set_fact(model.inputs()?[0], regular_input_fact.clone())?;
    batch_model.analyse()?;

    /*
    let mut analyser = stream_model.analyser()?
        .with_input_hint(input)?;
    analyser.analyse()?;
    */

    let mut display_graph = ::display_graph::DisplayGraph::from_model(&stream_model)?
        .with_graph_def(&params.graph)?;

    let eval_order = ::tract::model::eval_order(&stream_model)?;

    /*
    for mut dn in &mut display_graph.nodes {
        dn.hidden = true;
    }
    */

    // plan and state for reference batch mode
    let batch_plan = SimplePlan::new(&batch_model)?;
    let mut batch_state = SimpleState::new(&batch_plan)?;
    batch_state.set_input(0, fixed_input.clone())?;

    let mut failure = false;

    for node in eval_order.iter() {
        let node = &stream_model.nodes()[*node];
        //        println!("node: {:?}", node);

        if node.op.name() == "Source" || node.op.name() == "Const" {
            continue;
        }

        let batch_node = &batch_model.node_by_name(&node.name)?;
        batch_state.compute_recursively(batch_node.id)?;
        let batch_expected = &batch_state.values[batch_node.id].as_ref().unwrap()[0];
        let out_edge = &node.outputs[0];
        let out_edge_fact = &out_edge.fact;
        let out_stream_axis = out_edge_fact.stream_info()?.unwrap().axis;

        //         println!("expected: {:?}", batch_expected.shape());
        //         for line in batch_expected.as_f32s().unwrap().axis_chunks_iter(Axis(out_stream_axis), 1).take(10) {
        //             println!("  expected: {:?}", line.iter().take(5).cloned().collect::<Vec<f32>>());
        //         }
        //
        let mut batch_expected = batch_expected
            .as_f32s()
            .unwrap()
            .axis_chunks_iter(Axis(out_stream_axis), 1);

        // Run streaming node
        /*
        let facts = analyser.facts(node.id)?;
        let new_op = node.op.final_prep(facts.0, facts.1)?;

        let op = new_op.as_ref().unwrap_or(&node.op);
        */
        let mut buffers = node.op.new_buffer();

        /*
        let edges: Vec<_> = analyser.prev_edges[node.id]
            .iter()
            .map(|id| &analyser.edges[*id])
            .collect();
        */

        let mut input_offset = 0;
        let mut lines = vec![];
        let mut matched = 0;

        'stream: loop {
            let mut inputs = tvec!();
            for &edge in &node.inputs {
                if let Some(info) = stream_model.fact(edge)?.stream_info()? {
                    let prec_name = &stream_model.nodes()[edge.node].name;
                    let batch_prec_node = batch_state.model().node_by_name(&prec_name)?;
                    let data = &batch_state.values[batch_prec_node.id].as_ref().unwrap()[edge.slot];
                    let data = data.as_f32s().unwrap();
                    let chunk = data
                        .axis_chunks_iter(Axis(info.axis), 1)
                        .skip(input_offset)
                        .next()
                        .unwrap();
                    let stream = Stream {
                        info,
                        offset: input_offset as _,
                        chunk: Some(chunk.to_owned().into()),
                    };
                    inputs.push(StepValue::Stream(stream));
                } else {
                    let value = stream_model.nodes()[edge.node]
                        .op()
                        .const_value()
                        .ok_or("Not a const")?;
                    inputs.push(StepValue::Const(value.into()))
                }
                //                println!("input {:?}", inputs.last().unwrap());
            }
            let output = node.op.step(inputs, &mut buffers)?;
            input_offset += 1;
            let output = if let Some(output) = output {
                output
            } else {
                continue;
            };
            let found: &Tensor = &output[0];
            let found = found.as_f32s().unwrap();
            for found in found.axis_chunks_iter(Axis(out_stream_axis), 1) {
                let found: Tensor = found.to_owned().into();
                lines.push(format!("found: {:?}", found));
                if let Some(expected) = batch_expected.next() {
                    lines.push(format!("expected: {:?}", Tensor::from(expected.to_owned())));
                    lines.push("".into());
                    let expected = expected.to_owned().into();
                    if found.close_enough(&expected, node.op.rounding_errors()) {
                        matched += 1;
                        if matched == wanted_matches {
                            break 'stream;
                        }
                    } else {
                        //   println!("found: {:?}", found.as_f32s().unwrap().iter().take(5).join(" "));
                        //    break 'stream;
                    }
                } else {
                    break 'stream;
                }
            }
        }
        println!("matched : {}", matched);
        if matched == wanted_matches {
            display_graph.add_node_label(node.id, "OK".green().to_string())?;
        } else {
            display_graph.add_node_label(node.id, "MISMATCH".red().to_string())?;
            // dn.hidden = false;
            //            dn.more_lines.push(format!("matched {} records streaming dim {:?}", matched, out_edge_fact.stream_dim()?));
            //            dn.more_lines.extend(lines.into_iter());
            failure = true;
        }
    }

    if failure {
        display_graph.render(&output_params)?;
    } else if log_enabled!(Info) {
        display_graph.render(&output_params)?;
    } else {
        println!("{}", "Each node passed the comparison.".bold().green());
    }
    Ok(())
}

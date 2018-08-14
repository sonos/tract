use colored::Colorize;
use simplelog::Level::Info;
use ndarray::Axis;
use tfdeploy::analyser::TensorFact;
use tfdeploy::streaming::StreamingPlan;
use tfdeploy::tensor::Datum;
use tfdeploy::{SimplePlan, Tensor};
use {OutputParameters, Parameters};
use errors::*;

pub fn handle(params: Parameters, output_params: OutputParameters) -> Result<()> {
    let model = params.tfd_model;
    let input = params.input.as_ref().unwrap();

    // First generate random values for the inputs.
    let fixed_input = input.to_tensor_with_stream_dim(Some(500))?;

    let regular_input_fact = TensorFact::default()
        .with_shape(fixed_input.shape())
        .with_datum_type(fixed_input.datum_type());

    /*
    let optimized_model = model
        .analyser(&params.output_node)?
        .with_hint(&params.input_nodes[0], &regular_input_fact)?
        .to_optimized_model()?;
    */

    let stream_model = model
        .analyser(&params.output_node)?
        .with_hint(&params.input_nodes[0], &input.to_fact())?
        .to_optimized_model()?;

    let mut analyser = stream_model
        .analyser(&params.output_node)?
        .with_hint(&params.input_nodes[0], &input.to_fact())?;
    analyser.analyse()?;

    let mut display_graph =
        ::display_graph::DisplayGraph::from_nodes(&stream_model.nodes)?
        .with_analyser(&analyser)?;

    let eval_order = ::tfdeploy::model::eval_order_for_nodes(&stream_model.nodes(), &[stream_model.node_by_name(&params.output_node)?.id])?;

    for mut dn in &mut display_graph.nodes {
        dn.hidden = true;
    }

    let mut failure = false;

    for node in eval_order.iter() {
        let dn = &mut display_graph.nodes[*node];
        let node = &stream_model.nodes()[*node];

        if !model.node_by_name(&node.name).is_ok() || node.op_name == "Placeholder" {
            continue;
        }

        info!("Checking node {} {} output", node.id, node.name);

        // Run unmodified graph
        let original_plan = SimplePlan::new(&model, &params.input_nodes, &[&node.name])?;
        let original_output = original_plan.run(vec!(fixed_input.clone()))?;

        // Run streaming graph
        let streaming_plan = StreamingPlan::new(
            &stream_model,
            vec![(&params.input_nodes[0].as_ref(), input.to_fact())],
            Some(&node.name),
        )?;
        let mut state = streaming_plan.state()?;

        let output_streaming_dim = streaming_plan.output_streaming_dim()?;
        let mut expected_output = original_output[0][0]
            .as_f32s()
            .unwrap()
            .axis_chunks_iter(Axis(output_streaming_dim), 1);

        let values = fixed_input.as_f32s().unwrap();
        let streaming_dim = input.to_fact().streaming_dim()?;

        let mut matched = 0;
        let mut lines = vec!();
        for chunk in values.axis_chunks_iter(Axis(streaming_dim), 1) {
            let output = state.step(0, f32::array_into_tensor(chunk.to_owned()))?;
            if output.len() > 0 {
                let found: &Tensor = &output[0][0];
                lines.push(format!("found: {:?}", f32::tensor_to_view(found).unwrap()));
                if let Some(expected) = expected_output.next() {
                    lines.push(format!("expected: {:?}", expected));
                    let expected = expected.to_owned().into();
                    if found.close_enough(&expected) {
                        matched += 1;
                        if matched > 10 {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        if matched > 10 {
            dn.label = Some("OK".green().to_string());
        } else {
            dn.label = Some("MISMATCH".red().to_string());
            dn.hidden = false;
            dn.more_lines.push(format!("matched {} records streaming dim {}", matched, streaming_dim));
            dn.more_lines.extend(lines.into_iter());
            failure = true;
            break;
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

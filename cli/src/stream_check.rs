use errors::*;
use ndarray::Axis;
use tfdeploy::streaming::StreamingPlan;
use tfdeploy::tensor::Datum;
use tfdeploy::{SimplePlan, Tensor};
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, _output_params: OutputParameters) -> Result<()> {
    let model = params.tfd_model;
    let input = params.input.as_ref().unwrap();

    // First generate random values for the inputs.
    let fixed_input = vec![input.to_tensor_with_stream_dim(Some(500))?];

    // Run unmodified graph
    let original_plan = SimplePlan::new(&model, &params.input_nodes, &[&params.output_node])?;
    let original_output = original_plan.run(fixed_input.clone())?;

    let optimized_model = model
        .analyser(&params.output_node)?
        .with_hint(&params.input_nodes[0], &input.to_fact())?
        .to_optimized_model()?;

    // Run streaming graph
    let streaming_plan = StreamingPlan::new(
        &optimized_model,
        vec![(&params.input_nodes[0], input.to_fact())],
        Some(&params.output_node),
    )?;
    let mut state = streaming_plan.state()?;

    let output_streaming_dim = streaming_plan.output_streaming_dim()?;
    let mut expected_output = original_output[0][0]
        .as_f32s()
        .unwrap()
        .axis_chunks_iter(Axis(output_streaming_dim), 1);

    let values = fixed_input[0].as_f32s().unwrap();
    let streaming_dim = input.to_fact().streaming_dim()?;
    for chunk in values.axis_chunks_iter(Axis(streaming_dim), 1) {
        let output = state.step(0, f32::array_into_tensor(chunk.to_owned()))?;
        if output.len() > 0 {
            let found: &Tensor = &output[0][0];
            let expected: Tensor = expected_output.next().unwrap().to_owned().into();
            assert!(found.close_enough(&expected));
        }
    }
    assert!(expected_output.next().is_none());
    info!("Looks good!");
    Ok(())
}

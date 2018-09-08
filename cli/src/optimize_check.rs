use errors::*;
use tfdeploy::plan::SimplePlan;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, _output_params: OutputParameters) -> CliResult<()> {
    let model = params.tfd_model;

    // First generate random values for the inputs.
    let fixed_input = tvec![params.input.as_ref().unwrap().to_tensor()?];

    // Run unmodified graph
    let original_plan = SimplePlan::new(&model, &params.input_nodes, &[&params.output_node])?;
    let original_output = original_plan.run(fixed_input.clone())?;

    info!("Setting up analyser.");

    let mut analyser = model
        .analyser(&params.output_node)?
        .with_hint(&params.input_nodes[0], &params.input.unwrap().to_fact())?;

    info!("Running analyse");
    let optimized_model = analyser.to_optimized_model()?;
    info!(
        "Size of the graph after pruning: {:?} nodes.",
        optimized_model.nodes.len()
    );

    // Run optimized graph
    let optimized_plan =
        SimplePlan::new(&optimized_model, &params.input_nodes, &[params.output_node])?;
    let optimized_output = optimized_plan.run(fixed_input.clone())?;

    if original_output.len() != optimized_output.len() {
        bail!(
            "Output nodes count are different: original:{} optimized:{}",
            original_output.len(),
            optimized_output.len()
        )
    }
    for (a, b) in original_output.iter().zip(optimized_output.iter()) {
        if a.len() != b.len() {
            bail!(
                "Output node tensor counts are different. original:{}, optimized:{}",
                a.len(),
                b.len()
            )
        }
        for (a, b) in a.iter().zip(b.iter()) {
            if !a.close_enough(b, true) {
                bail!("Different output {:?} and {:?}", a, b)
            }
        }
    }
    info!("Looks good!");
    Ok(())
}

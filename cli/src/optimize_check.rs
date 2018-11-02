use errors::*;
use tract_core::plan::SimplePlan;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, _output_params: OutputParameters) -> CliResult<()> {
    let model = &params.tfd_model;

    // First generate random values for the inputs.
    let fixed_inputs = ::tensor::make_inputs(&[params.tfd_model.input_fact()?.clone()])?;

    // Run unmodified graph
    let original_plan = SimplePlan::new(model)?;
    let original_output = original_plan.run(fixed_inputs.clone())?;

    info!("Running analyse");
    unimplemented!();
    /*
    model.analyse();
    let optimized_model = analyser.to_optimized_model()?;
    info!(
        "Size of the graph after pruning: {:?} nodes.",
        optimized_model.nodes().len()
    );

    // Run optimized graph
    let optimized_plan = SimplePlan::for_model(&optimized_model)?;
    let optimized_output = optimized_plan.run(fixed_inputs.clone())?;

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
    */
    info!("Looks good!");
    Ok(())
}

use tfdeploy::SimplePlan;
use tfdeploy::analyser::TensorFact;
use errors::*;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, assert_outputs:Option<Vec<TensorFact>>, _output_params: OutputParameters) -> CliResult<()> {
    let tfd = params.tfd_model;

    let plan = SimplePlan::new(&tfd)?;
    let outputs = plan.run(params.inputs.unwrap().iter().map(|t| t.clone().unwrap()).collect())?;

    if let Some(asserts) = assert_outputs {
        ::utils::check_outputs(&outputs, &asserts)?;
    }

    Ok(())
}

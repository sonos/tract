use errors::*;
use tfdeploy::analyser::TensorFact;
use tfdeploy::SimplePlan;
use Parameters;

pub fn handle(params: Parameters, assert_outputs: Option<Vec<TensorFact>>) -> CliResult<()> {
    let tfd = params.tfd_model;

    let plan = SimplePlan::new(&tfd)?;
    let outputs = plan.run(
        params
            .inputs
            .unwrap()
            .iter()
            .map(|t| t.clone().unwrap())
            .collect(),
    )?;

    for (ix, output) in outputs.iter().enumerate() {
        println!("output #{}\n{}\n", ix, output.dump(true)?);
    }

    if let Some(asserts) = assert_outputs {
        ::utils::check_outputs(&outputs, &asserts)?;
    }

    Ok(())
}

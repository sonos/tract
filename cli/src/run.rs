use tfdeploy::SimplePlan;
use tfdeploy::analyser::TensorFact;
use display_graph::DisplayGraph;
use errors::*;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, assert_outputs:Option<Vec<TensorFact>>, output_params: OutputParameters) -> CliResult<()> {
    let tfd = params.tfd_model;

//    let plan = ::tfdeploy::model::eval_order(&tfd)?;

    let display_graph = DisplayGraph::from_model(&tfd)?.with_graph_def(&params.graph)?;
    display_graph.render(&output_params)?;

    let plan = SimplePlan::new(&tfd)?;
    let outputs = plan.run(params.inputs.unwrap().iter().map(|t| t.clone().unwrap()).collect())?;

    if let Some(asserts) = assert_outputs {
        ::utils::check_outputs(&outputs, &asserts)?;
    }

    Ok(())
}

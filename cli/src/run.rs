use tfdeploy::SimplePlan;
use tfdeploy::analyser::Fact;
use tfdeploy::analyser::TensorFact;
use display_graph::DisplayGraph;
use errors::*;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, assert_outputs:Option<Vec<TensorFact>>, output_params: OutputParameters) -> CliResult<()> {
    let tfd = params.tfd_model;
    let output_id = tfd.outputs()?[0].id;

    let plan = ::tfdeploy::model::eval_order_for_nodes(&tfd.nodes(), &[output_id])?;

    let nodes: Vec<_> = plan.iter().map(|i| &tfd.nodes()[*i]).collect();
    let display_graph = DisplayGraph::from_nodes(&*nodes)?.with_graph_def(&params.graph)?;
    display_graph.render(&output_params)?;

    let plan = SimplePlan::for_model(&tfd)?;
    let outputs = plan.run(params.inputs.iter().map(|tf| tf.concretize().unwrap()).collect())?;

    if let Some(asserts) = assert_outputs {
        ::utils::check_outputs(&outputs[0], &asserts)?;
    }

    Ok(())
}

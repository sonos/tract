use display_graph::DisplayGraph;
use errors::*;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, output_params: OutputParameters) -> CliResult<()> {
    let tfd = params.tfd_model;
    let output_id = tfd.outputs()?[0].id;
    let plan = ::tfdeploy::model::eval_order_for_nodes(&tfd.nodes(), &[output_id])?;

    let nodes: Vec<_> = plan.iter().map(|i| &tfd.nodes()[*i]).collect();
    let display_graph = DisplayGraph::from_nodes(&*nodes)?.with_graph_def(&params.graph)?;
    display_graph.render(&output_params)?;

    Ok(())
}

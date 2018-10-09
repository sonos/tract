use display_graph::DisplayGraph;
use errors::*;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, output_params: OutputParameters) -> CliResult<()> {
    let tfd = params.tfd_model;
    let plan = ::tfdeploy::model::eval_order(&tfd)?;

    let display_graph = DisplayGraph::from_model(&tfd)?.with_graph_def(&params.graph)?;
    display_graph.render(&output_params)?;

    Ok(())
}

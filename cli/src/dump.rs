use display_graph::DisplayGraph;
use errors::*;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, output_params: OutputParameters) -> Result<()> {
    let tfd = params.tfd_model;
    let output = tfd.get_node_by_id(params.output_node_id)?;
    let plan = output.eval_order(&tfd)?;

    let nodes: Vec<_> = plan.iter().map(|i| &tfd.nodes[*i]).collect();
    let display_graph = DisplayGraph::from_nodes(&*nodes)?.with_graph_def(&params.graph)?;
    display_graph.render(&output_params)?;

    Ok(())
}

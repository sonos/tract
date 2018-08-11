use errors::*;
use {OutputParameters, Parameters};

/// Handles the `analyse` subcommand.
pub fn handle(params: Parameters, optimize: bool, output_params: OutputParameters) -> Result<()> {
    let model = params.tfd_model;

    let mut analyser = model.analyser(&params.output_node)?;

    // Add hints for the input nodes.
    if let Some(input) = params.input {
        analyser.hint(&params.input_nodes[0], &input.to_fact())?;
    }

    info!("Running analyse");
    let analyse_result = analyser.analyse();

    if analyse_result.is_ok() && optimize {
        info!(
            "Size of the graph before pruning: approx. {:.2?} Ko for {:?} nodes.",
            ::bincode::serialize(&analyser.nodes)?.len() as f64 * 1e-3,
            analyser.nodes.len()
        );

        let model = analyser.to_optimized_model()?;

        info!(
            "Size of the graph after pruning: approx. {:.2?} Ko for {:?} nodes.",
            ::bincode::serialize(&model.nodes)?.len() as f64 * 1e-3,
            model.nodes.len()
        );
    }

    let nodes: Vec<_> = analyser.nodes.iter().collect();
    let display = ::display_graph::DisplayGraph::from_nodes(&*nodes)?
        .with_graph_def(&params.graph)?
        .with_analyser(&analyser)?;
    display.render(&output_params)?;

    Ok(analyse_result?)
}

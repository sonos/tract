use errors::*;
use Parameters;

/// Handles the `prune` subcommand.
#[allow(dead_code)]
pub fn handle(params: Parameters) -> Result<()> {
    let model = params.tfd_model;

    info!("Starting the analysis.");

    let mut analyser = model.analyser(&params.output_node)?;

    info!(
        "Starting size of the graph: approx. {:?} bytes for {:?} nodes.",
        format!("{:?}", analyser.nodes).into_bytes().len(),
        analyser.nodes.len()
    );

    let optimized = analyser.to_optimized_model()?;

    info!(
        "Ending size of the graph: approx. {:?} bytes for {:?} nodes.",
        format!("{:?}", optimized.nodes).into_bytes().len(),
        optimized.nodes.len()
    );

    Ok(())
}

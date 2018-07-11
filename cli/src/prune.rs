use Parameters;
use errors::*;

use tfdeploy::analyser::Analyser;

/// Handles the `prune` subcommand.
#[allow(dead_code)]
pub fn handle(params: Parameters) -> Result<()> {
    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output_node_id)?.id;

    info!("Starting the analysis.");

    let mut analyser = Analyser::new(model, output)?;

    info!(
        "Starting size of the graph: approx. {:?} bytes for {:?} nodes.",
        format!("{:?}", analyser.nodes).into_bytes().len(),
        analyser.nodes.len()
    );

    analyser.run()?;
    analyser.propagate_constants()?;
    analyser.prune_unused();

    info!(
        "Ending size of the graph: approx. {:?} bytes for {:?} nodes.",
        format!("{:?}", analyser.nodes).into_bytes().len(),
        analyser.nodes.len()
    );

    Ok(())
}

use Parameters;
use errors::*;

use std::fs::File;
use std::io::prelude::*;

use tfdeploy::analyser::Analyser;
use tfdeploy::analyser::{TensorFact, ShapeFact, DimFact};

/// Handles the `analyse` subcommand.
pub fn handle(params: Parameters, prune: bool, web: bool) -> Result<()> {

    let model = params.tfd_model;
    let output = model.get_node_by_id(params.output_node_id)?.id;

    info!("Setting up analyser.");

    let mut analyser = Analyser::new(model, output)?;

    // Add hints for the input nodes.
    if let Some(input) = params.input {
        let dims = input.shape.iter()
            .map(|d| match d {
                None    => DimFact::Streamed,
                Some(i) => DimFact::Only(*i),
            })
            .collect::<Vec<_>>();

        for &i in &params.input_node_ids {
            analyser.hint(i, &TensorFact {
                datatype: typefact!(input.datatype),
                shape: ShapeFact::closed(dims.clone()),
                value: valuefact!(_),
            })?;
        }
    }

    info!("Running analyse");
    analyser.run()?;

    // Prune constant nodes if needed.
    if prune {
        info!(
            "Size of the graph before pruning: approx. {:.2?} Ko for {:?} nodes.",
            ::bincode::serialize(&analyser.nodes)?.len() as f64 * 1e-3,
            analyser.nodes.len()
        );

        analyser.propagate_constants()?;
        analyser.prune_unused();

        info!(
            "Size of the graph after pruning: approx. {:.2?} Ko for {:?} nodes.",
            ::bincode::serialize(&analyser.nodes)?.len() as f64 * 1e-3,
            analyser.nodes.len()
        );
    }


    // Display an interactive view of the graph if needed.
    let data = ::serde_json::to_vec(&(&analyser.nodes, &analyser.edges)).unwrap();
    if web {
        ::web::open_web(data);
    } else {
        File::create("analyser.json")?.write_all(&data)?;
        println!("Wrote the result of the analysis to analyser.json.");
    }

    Ok(())
}


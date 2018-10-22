use display_graph::*;
use errors::*;
use tfdeploy::ops::prelude::*;
use Parameters;

pub fn handle(
    params: Parameters,
    assert_outputs: Option<Vec<TensorFact>>,
    options: DisplayOptions,
) -> CliResult<()> {
    let tfd = if let Some(ref pulsed) = params.pulsified_model.as_ref() {
        &pulsed.model
    } else {
        &params.tfd_model
    };

    let mut display_graph =
        DisplayGraph::from_model_and_options(tfd, options)?.with_graph_def(&params.graph)?;
    if let Some(ref pulsed) = params.pulsified_model.as_ref() {
        for (i, fact) in pulsed.facts.iter() {
            display_graph
                .add_node_label(i.node, format!("Output pulse: {} fact: {:?}", i.slot, fact));
        }
    };
    display_graph.render()?;

    if let Some(asserts) = assert_outputs {
        for (ix, assert) in asserts.iter().enumerate() {
            assert.unify(tfd.outputs_fact(ix).unwrap())?;
        }
    }

    Ok(())
}

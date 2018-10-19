use display_graph::*;
use errors::*;
use tfdeploy::ops::prelude::*;
use Parameters;

pub fn handle(
    params: Parameters,
    assert_outputs: Option<Vec<TensorFact>>,
    options: DisplayOptions,
) -> CliResult<()> {
    let tfd = params.tfd_model;

    let display_graph =
        DisplayGraph::from_model_and_options(&tfd, options)?.with_graph_def(&params.graph)?;
    display_graph.render()?;

    if let Some(asserts) = assert_outputs {
        for (ix, assert) in asserts.iter().enumerate() {
            assert.unify(tfd.outputs_fact(ix).unwrap())?;
        }
    }

    Ok(())
}

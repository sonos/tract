use display_graph::DisplayGraph;
use errors::*;
use tfdeploy::analyser::TensorFact;
use tfdeploy::analyser::Fact;
use {OutputParameters, Parameters};

pub fn handle(params: Parameters, assert_outputs:Option<Vec<TensorFact>>, output_params: OutputParameters) -> CliResult<()> {
    let tfd = params.tfd_model;

    let display_graph = DisplayGraph::from_model(&tfd)?.with_graph_def(&params.graph)?;
    display_graph.render(&output_params)?;

    if let Some(asserts) = assert_outputs {
        for (ix, assert) in asserts.iter().enumerate() {
            assert.unify(tfd.outputs_fact(ix).unwrap())?;
        }
    }

    Ok(())
}

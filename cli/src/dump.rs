use crate::display_graph::*;
use crate::errors::*;
use crate::Parameters;
use tract_core::ops::prelude::*;

pub fn handle(params: Parameters, options: DisplayOptions) -> CliResult<()> {
    let tract = &params.tract_model;

    let display_graph =
        DisplayGraph::from_model_and_options(tract, options)?.with_graph_def(&params.graph)?;
    display_graph.render()?;

    if let Some(asserts) = params.assertions {
        if let Some(asserts) = asserts.assert_outputs {
            for (ix, assert) in asserts.iter().enumerate() {
                assert.unify(tract.outputs_fact(ix).unwrap())?;
            }
        }
        if let Some(asserts) = asserts.assert_output_facts {
            let outputs_facts: Vec<TensorFact> = tract
                .outputs()?
                .iter()
                .map(|o| tract.fact(*o).unwrap().clone())
                .collect();
            crate::utils::check_inferred(&*outputs_facts, &*asserts)?;
        }
    }

    Ok(())
}

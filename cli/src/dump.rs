use crate::display_graph::*;
use crate::errors::*;
use crate::{Parameters, SomeModel};
use tract_core::ops::prelude::*;

pub fn handle(params: Parameters, options: DisplayOptions) -> CliResult<()> {
    let tract = &params.tract_model;
    match tract {
        SomeModel::Inference(m) => handle_t(m, &params, options),
        SomeModel::Typed(m) => handle_t(m, &params, options),
        SomeModel::Normalized(m) => handle_t(m, &params, options),
        SomeModel::Pulsed(_, m) => handle_t(m, &params, options),
    }
}

fn handle_t<TI: TensorInfo>(
    tract: &Model<TI>,
    params: &Parameters,
    options: DisplayOptions,
) -> CliResult<()> {
    let display_graph =
        DisplayGraph::from_model_and_options(tract, options)?.with_graph_def(&params.graph)?;
    display_graph.render()?;

    if let Some(asserts) = &params.assertions {
        if let Some(asserts) = &asserts.assert_outputs {
            for (ix, assert) in asserts.iter().enumerate() {
                assert.unify(&tract.outputs_fact(ix).unwrap().to_tensor_fact())?;
            }
        }
        if let Some(asserts) = &asserts.assert_output_facts {
            let outputs_facts: Vec<TensorFact> =
                tract.outputs()?.iter().map(|o| tract.fact(*o).unwrap().to_tensor_fact()).collect();
            crate::utils::check_inferred(&*outputs_facts, &*asserts)?;
        }
    }

    Ok(())
}

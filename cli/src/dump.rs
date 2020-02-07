use crate::display_graph::*;
use crate::errors::*;
use crate::{Model, Parameters};
use tract_core::internal::*;

pub fn handle(params: &Parameters, options: DisplayOptions, _inner: Vec<String>) -> CliResult<()> {
    let tract = &params.tract_model;
    handle_model(&**tract, &params, options)
}

pub fn handle_model(
    model: &dyn Model,
    params: &Parameters,
    options: DisplayOptions,
) -> CliResult<()> {
    let display_graph = DisplayGraph::from_model_and_options(model, Arc::new(options))?
        .with_graph_def(&params.graph)?;
    display_graph.render()?;

    if let Some(asserts) = &params.assertions {
        if let Some(asserts) = &asserts.assert_output_facts {
            let outputs_facts: Vec<InferenceFact> = model
                .output_outlets()
                .iter()
                .map(|o| Ok(InferenceFact::from(&model.outlet_typedfact(*o)?)))
                .collect::<TractResult<Vec<InferenceFact>>>()?;
            crate::utils::check_inferred(&*outputs_facts, &*asserts)?;
        }
    }

    Ok(())
}

/*
fn handle_inner(tract: &TypedModel, params: &Parameters, options: DisplayOptions, inner: Vec<String>) -> CliResult<()> {
    if let Some(node) = inner.get(0) {
        let node = tract.node(node.parse()?);
        if let Some(scan) = node.op_as::<tract_core::ops::rec::scan::Scan<TypedFact, Box<Op>>>() {
            let model = &scan.body;
            handle_model(model, params, options)?
        }
    }
    Ok(())
}
*/

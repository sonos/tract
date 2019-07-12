use crate::display_graph::*;
use crate::errors::*;
use crate::{Parameters, SomeModel};
use std::fmt::{Debug, Display};
use tract_core::internal::*;

pub fn handle(params: Parameters, options: DisplayOptions, inner:Vec<String>) -> CliResult<()> {
    let tract = &params.tract_model;
    match tract {
        SomeModel::Inference(m) => handle_t(m, &params, options),
        SomeModel::Typed(m) if inner.len() == 0 => handle_t(m, &params, options),
        SomeModel::Typed(m)  => handle_inner(m, &params, options, inner),
        SomeModel::Normalized(m) => handle_t(m, &params, options),
        SomeModel::Pulsed(m) => handle_t(m, &params, options),
    }
}

fn handle_t<TI, O>(
    tract: &Model<TI, O>,
    params: &Parameters,
    options: DisplayOptions,
) -> CliResult<()>
where
    TI: TensorInfo,
    O: AsRef<Op> + AsMut<Op> + Display + Debug,
{
    let display_graph =
        DisplayGraph::from_model_and_options(tract, options)?.with_graph_def(&params.graph)?;
    display_graph.render()?;

    if let Some(asserts) = &params.assertions {
        if let Some(asserts) = &asserts.assert_output_facts {
            let outputs_facts: Vec<TensorFact> = tract
                .output_outlets()?
                .iter()
                .map(|o| tract.outlet_fact(*o).unwrap().to_tensor_fact())
                .collect();
            crate::utils::check_inferred(&*outputs_facts, &*asserts)?;
        }
    }

    Ok(())
}

fn handle_inner(tract: &TypedModel, params: &Parameters, options: DisplayOptions, inner: Vec<String>) -> CliResult<()> {
    if let Some(node) = inner.get(0) {
        let node = tract.node(node.parse()?);
        if let Some(scan) = node.op_as::<tract_core::ops::rec::scan::Scan<TypedTensorInfo, Box<Op>>>() {
            let model = &scan.body;
            handle_t(model, params, options)?
        }
    }
    Ok(())
}

use crate::display_graph::*;
use crate::errors::*;
use crate::{Model, Parameters};
use tract_core::internal::*;

pub fn handle(params: Parameters, options: DisplayOptions) -> CliResult<()> {
    let tract = &params.tract_model;
    if let Some(_) = tract.downcast_ref::<InferenceModel>() {
        bail!("Cost eval only work on a typd model")
    } else if let Some(m) = tract.downcast_ref::<TypedModel>() {
        handle_t(m, &params, options)
    } else if let Some(m) = tract.downcast_ref::<NormalizedModel>() {
        handle_t(&m.clone().into_typed()?, &params, options)
    } else if let Some(m) = tract.downcast_ref::<PulsedModel>() {
        handle_t(&m.clone().into_typed()?, &params, options)
    } else {
        bail!("Pulse model are unsupported here")
    }
}

fn handle_t(model: &TypedModel, params: &Parameters, options: DisplayOptions) -> CliResult<()> {
    let mut total: HashMap<Cost, TDim> = HashMap::default();
    let mut display_graph =
        DisplayGraph::from_model_and_options(model as &dyn Model, options.into())?
            .with_graph_def(&params.graph)?;
    for i in ::tract_core::model::eval_order(&model)? {
        let inputs = model.node_input_facts(i)?;
        let cost = model.nodes()[i].op().cost(&*inputs)?;
        if !cost.is_empty() {
            let rows = cost
                .iter()
                .inspect(|(c, i)| *total.entry(*c).or_insert(0.to_dim()) += i)
                .map(|(c, i)| format!("{:?} {:?}", c, i))
                .collect();
            display_graph.add_node_section(i, rows)?;
        }
    }
    display_graph.render()?;
    for (c, i) in total {
        println!("{:?}: {:?}", c, i);
    }
    Ok(())
}

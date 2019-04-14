use crate::format::Row;
use crate::display_graph::*;
use crate::errors::*;
use crate::{Parameters, SomeModel};
use tract_core::internal::*;

pub fn handle(params: Parameters, options: DisplayOptions) -> CliResult<()> {
    let tract = &params.tract_model;
    match tract {
        SomeModel::Inference(_) => panic!("Cost can only be performed on typed nets"),
        SomeModel::Typed(m) => handle_t(m, &params, options),
        SomeModel::Normalized(m) => handle_t(&m.clone().into_typed()?, &params, options),
        SomeModel::Pulsed(_, m) => handle_t(&m.clone().into_typed()?, &params, options),
    }
}

fn handle_t(
    model: &Model<TypedTensorInfo>,
    params: &Parameters,
    options: DisplayOptions,
) -> CliResult<()> {
    let mut total:HashMap<Cost,TDim> = HashMap::default();
    let mut display_graph =
        DisplayGraph::from_model_and_options(model, options)?.with_graph_def(&params.graph)?;
    for i in ::tract_core::model::eval_order(&model)? {
        let inputs = model.node_input_facts(i)?;
        let cost = model.nodes()[i].op().cost(&*inputs)?;
        if !cost.is_empty() {
            let rows = cost
                .iter()
                .inspect(|(c,i)| *total.entry(*c).or_insert(0.to_dim()) += *i)
                .map(|(c, i)| Row::Double(format!("{:?}", c), format!("{:?}", i)))
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

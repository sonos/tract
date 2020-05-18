use crate::display_graph::*;
use crate::display_params::*;
use crate::errors::*;
use crate::{BenchLimits, Parameters};
use tract_hir::internal::*;

pub fn handle(
    params: &Parameters,
    options: &DisplayParams,
    matches: &clap::ArgMatches,
    bench_limits: &BenchLimits,
    _inner: Vec<String>,
) -> CliResult<()> {
    let model = &params.tract_model;
    let mut display_graph = DisplayGraph::from_model_and_options(model.as_ref(), options)?
        .with_graph_def(model.as_ref(), &params.graph)?;
    if options.profile {
        let model = params
            .tract_model
            .downcast_ref::<TypedModel>()
            .ok_or("Can only profile typed models")?;
        crate::profile::profile(model, bench_limits, &mut display_graph)?;
    }

    display_graph.render(model.as_ref())?;

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

    /*
    if options.cost {
        let total: HashMap<Cost, TDim> = display_graph.total_cost()?;
        println!("{}", White.bold().paint("Cost summary"));
        for (c, i) in &total {
            println!(" * {:?}: {}", c, i);
        }
        let assert = matches.value_of("assert-cost").map(|a| crate::cost::parse_costs(a));
        if let Some(assert) = assert {
            let assert: HashMap<Cost, TDim> =
                assert.iter().map(|(c, n)| (*c, n.to_dim())).collect();
            if assert != total {
                bail!("Cost assertion not met: expected {:?} got {:?}", assert, total);
            }
        }
    }

    if let Some(profile) = display_graph.profile_data.as_ref() {
        println!("{}", White.bold().paint("Most time consuming operations"));
        for (op, (dur, n)) in profile
            .by_ops(model.as_ref())?
            .into_iter()
            .sorted_by(|(_, (a, _)), (_, (b, _))| b.cmp(&a))
        {
            println!(
                " * {} {:3} nodes: {}",
                Blue.bold().paint(format!("{:20}", op)),
                n,
                crate::format::dur_avg_ratio(dur, profile.sum)
            );
        }
    }
    */

    Ok(())
}

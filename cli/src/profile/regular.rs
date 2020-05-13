use std::fmt::{Debug, Display};

use ansi_term::Colour::*;
use log::Level::Info;

use crate::display_graph::DisplayOptions;
use crate::errors::*;
use crate::{Model, Parameters, ProfilingMode};

use crate::format::*;
use crate::profile::ProfileData;
use crate::tensor::make_inputs;
use std::time::{Duration, Instant};

use tract_hir::internal::*;

use readings_probe::*;

pub fn handle_benching(
    params: &Parameters,
    profiling: ProfilingMode,
    probe: Option<&Probe>,
) -> CliResult<()> {
    dispatch_model!(params.tract_model, |m| handle_benching_t(m, &params, profiling, probe))
}

pub fn make_inputs_for_model<F, O>(model: &ModelImpl<F, O>) -> CliResult<TVec<Tensor>>
where
    F: Fact + Clone + 'static + Hash,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static + Hash,
{
    Ok(make_inputs(
        &*model
            .input_outlets()?
            .iter()
            .map(|&t| model.outlet_typedfact(t))
            .collect::<TractResult<Vec<TypedFact>>>()?,
    )?)
}

fn handle_benching_t<F, O>(
    model: &ModelImpl<F, O>,
    params: &Parameters,
    profiling: ProfilingMode,
    probe: Option<&Probe>,
) -> CliResult<()>
where
    F: Fact + Clone + 'static + Hash,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static + Hash,
{
    let (max_iters, max_time) =
        if let ProfilingMode::RegularBenching { max_iters, max_time } = profiling {
            (max_iters, max_time)
        } else {
            bail!("Expecting bench profile mode")
        };

    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(plan)?;
    let progress = probe.and_then(|m| m.get_i64("progress"));
    info!("Starting bench itself");
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed() < max_time {
        if let Some(mon) = probe {
            let _ = mon.log_event(&format!("loop_{}", iters));
        }
        if let Some(p) = &progress {
            p.store(iters as _, std::sync::atomic::Ordering::Relaxed);
        }
        state.run(make_inputs_for_model(model)?)?;
        iters += 1;
    }
    let dur = start.elapsed();
    let dur = Duration::from_secs_f64(dur.as_secs_f64() / iters as f64);

    if params.machine_friendly {
        println!("real: {}", dur.as_secs_f64());
    } else {
        println!("Bench ran {} times.\n{}", iters, dur_avg_multiline(dur));
    }

    Ok(())
}

pub fn handle(
    params: &Parameters,
    profiling: ProfilingMode,
    display_options: DisplayOptions,
) -> CliResult<()> {
    dispatch_model!(params.tract_model, |m| handle_t(m, &params, profiling, display_options))
}

/// Handles the `profile` subcommand when there are no streaming dimensions.
pub fn handle_t<F, O>(
    model: &ModelImpl<F, O>,
    params: &Parameters,
    profiling: ProfilingMode,
    mut display_options: DisplayOptions,
) -> CliResult<()>
where
    F: Fact + Clone + 'static + Hash,
    O: AsRef<dyn Op> + AsMut<dyn Op> + Display + Debug + Clone + 'static + Hash,
    ModelImpl<F, O>: Model,
{
    let (max_iters, max_time) = if let ProfilingMode::Regular { max_iters, max_time } = profiling {
        (max_iters, max_time)
    } else {
        bail!("Expecting regular profile mode")
    };

    let mut profile = ProfileData::default();

    info!("Running entire network");
    let plan = SimplePlan::new(model)?;
    let mut state = SimpleState::new(&plan)?;
    let mut iters = 0;
    let start = Instant::now();
    while iters < max_iters && start.elapsed() < max_time {
        let _ = state.run_plan_with_eval(
            make_inputs_for_model(model)?,
            0,
            |session_state, state, node, input| {
                let start = Instant::now();
                let r = tract_core::plan::eval(session_state, state, node, input);
                let elapsed = start.elapsed();
                profile.add([node.id].as_ref(), elapsed)?;
                r
            },
        )?;
        iters += 1;
    }
    let entire = Duration::from_secs_f64(start.elapsed().as_secs_f64() / iters as f64);

    info!("Running {} iterations max. for each node.", max_iters);
    info!("Running for {} ms max. for each node.", max_time.as_millis());

    for &outer_node in &plan.order {
        if let Some(m) =
            (model as &dyn Model).downcast_ref::<ModelImpl<TypedFact, Box<dyn TypedOp>>>()
        {
            let outer_node = m.node(outer_node);
            let inputs: TVec<TypedFact> = model
                .node_input_facts(outer_node.id)?
                .iter()
                .map(|&i| i.to_typed_fact())
                .collect::<TractResult<_>>()?;
            let ref_inputs: TVec<&TypedFact> = inputs.iter().collect();
            for ((_name, inner_model, _, _), (_name_, multiplier)) in outer_node
                .op
                .nested_models()
                .iter()
                .zip(outer_node.op.nested_model_multipliers(&ref_inputs).iter())
            {
                if let Some(inner_model) = inner_model.downcast_ref::<TypedModel>() {
                    for _ in 0..iters {
                        let inner_plan = SimplePlan::new(inner_model)?;
                        let mut state = SimpleState::new(inner_plan)?;
                        let _ = state.run_plan_with_eval(
                            make_inputs_for_model(inner_model)?,
                            0,
                            |session_state, state, node, input| {
                                let start = Instant::now();
                                let r = tract_core::plan::eval(session_state, state, node, input);
                                let elapsed = start.elapsed();
                                let elapsed = Duration::from_secs_f64(
                                    elapsed.as_secs_f64() * *multiplier as f64,
                                );
                                profile.add([outer_node.id, node.id].as_ref(), elapsed)?;
                                profile.sub([outer_node.id].as_ref(), elapsed)?;
                                r
                            },
                        )?;
                    }
                }
            }
        }
    }

    profile.scale((iters as f64).recip());

    if display_options == DisplayOptions::default() {
        display_options.node_ids = Some(profile.most_consuming_nodes()?);
    };

    let mut display_graph = crate::display_graph::DisplayGraph::from_model_and_options(
        model,
        Arc::new(display_options),
    )?
    .with_graph_def(&params.graph)?;

    let sum = profile.summed();
    for (ix, measure) in profile.nodes.iter() {
        display_graph.add_node_label(&ix, dur_avg_oneline_ratio(*measure, sum))?;
    }

    display_graph.render()?;
    println!();

    profile.print_most_consuming_ops(model)?;

    println!("Entire network performance: {}", dur_avg_oneline(entire));
    println!("Accounted by ops: {}", dur_avg_oneline_ratio(profile.summed(), entire));

    if log_enabled!(Info) {
        println!(
            "(Real: {} in total, with max_iters={:e} and max_time={:?}ms.)",
            White.paint(format!("{:.3} ms", profile.summed().as_secs_f64() * 1e3)),
            max_iters as f32,
            max_time,
        );
    }

    Ok(())
}

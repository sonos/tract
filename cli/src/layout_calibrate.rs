//! `tract <model> -O layout-calibrate --table <file>`: per-op layout costs of the channel-last depthwise flip, measured
//! in the model.
//!
//! The model is built twice, once as declutter leaves it and once with `TRACT_DISABLE_DEPTHWISE_CHANNEL_LAST`, and both
//! optimized plans are profiled. Nodes present under the same name in both plans are paired and grouped by the op key
//! of `tract_core::optim::layout_plan::node_key` in the channel-first plan; for each key the table accumulates the
//! channel-first time, the channel-last time and the channel-first work. An existing table file is added to, so running
//! the command over a set of models builds one table for the layout planner.
//!
//! The two plans only differ on hosts where declutter flips depthwise convolutions, i.e. that have a channel-last
//! depthwise kernel.
//!
//! `tract <model> -O layout-predict --table <file>` estimates, without running the model, the time change of flipping
//! every depthwise convolution, and with `--measure` profiles the channel-first, channel-last and planned plans.

use std::collections::HashMap;

use tract_core::internal::*;
use tract_core::optim::layout_plan::LayoutTable;
use tract_core::optim::layout_plan::estimate;
use tract_core::optim::layout_plan::node_key;
use tract_core::optim::layout_plan::signatures;
use tract_libcli::annotations::Annotations;
use tract_libcli::profile::profile;
use tract_libcli::tensor::get_or_make_inputs;

use crate::params::Parameters;
use crate::params::bench_limits_from_clap;
use crate::tensor::run_params_from_subcommand;
use readings_probe::Probe;

pub fn command() -> clap::Command {
    clap::Command::new("layout-calibrate")
        .about("Profile a model with and without the channel-last depthwise flip and add per-op layout costs to a table")
        .arg(clap::Arg::new("table").long("table").num_args(1).required(true).help("Table file, added to if it exists"))
}

pub fn predict_command() -> clap::Command {
    clap::Command::new("layout-predict")
        .about("Estimate the time change of the channel-last depthwise flip from a layout table")
        .arg(
            clap::Arg::new("table")
                .long("table")
                .num_args(1)
                .required(true)
                .help("Table written by layout-calibrate"),
        )
        .arg(
            clap::Arg::new("measure")
                .long("measure")
                .action(clap::ArgAction::SetTrue)
                .help("Also profile the channel-first, channel-last and planned plans"),
        )
}

fn typed(params: &Parameters) -> TractResult<&TypedModel> {
    params.tract_model.downcast_ref::<TypedModel>().context("Needs a typed model (pass -O)")
}

fn rebuild(
    matches: &clap::ArgMatches,
    probe: Option<&Probe>,
    knob: &str,
) -> TractResult<Parameters> {
    tract_core::knobs::KnobScope::new([(knob, "true")])?
        .enter(|| Parameters::from_clap(matches, probe))
}

/// Per node name, the profiled time of the optimized plan in `params`.
fn profile_nodes(params: &Parameters, sub: &clap::ArgMatches) -> TractResult<HashMap<String, f64>> {
    let model = typed(params)?;
    let runnable = params.req_runnable().context("Needs an optimized model (pass -O)")?;
    let run_params = run_params_from_subcommand(params, sub)?;
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;
    let mut annotations = Annotations::from_model(&*params.tract_model)?;
    profile(&runnable, &bench_limits_from_clap(sub)?, &mut annotations, &inputs, None, false)?;
    Ok(model
        .nodes()
        .iter()
        .filter_map(|n| {
            annotations
                .tags
                .get(&n.id.into())
                .and_then(|t| t.profile)
                .map(|d| (n.name.clone(), d.as_secs_f64()))
        })
        .collect())
}

/// `params` is the model as declutter leaves it; the channel-first plan is rebuilt from `matches` with the depthwise
/// channel-last suggestion disabled.
pub fn handle(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub: &clap::ArgMatches,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let table_path = sub.get_one::<String>("table").unwrap();
    let cf_params = rebuild(matches, probe, "TRACT_DISABLE_DEPTHWISE_CHANNEL_LAST")?;
    let (cl_model, cf_model) = (typed(params)?, typed(&cf_params)?);
    let cl_secs = profile_nodes(params, sub)?;
    let cf_secs = profile_nodes(&cf_params, sub)?;
    let mut table = match std::fs::read_to_string(table_path) {
        Ok(s) => LayoutTable::parse(&s).with_context(|| format!("Reading table {table_path}"))?,
        Err(_) => LayoutTable::default(),
    };
    table.models.push(matches.get_one::<String>("model").cloned().unwrap_or_default());
    let (cf_sig, cl_sig) = (signatures(cf_model), signatures(cl_model));
    let (mut paired, mut changed) = (0, 0);
    for node in cf_model.nodes() {
        let (Some(&cf), Some(&cl)) = (cf_secs.get(&node.name), cl_secs.get(&node.name)) else {
            continue;
        };
        paired += 1;
        changed += (cf_sig.get(&node.name) != cl_sig.get(&node.name)) as usize;
        let (key, unit, work) = node_key(cf_model, node)?;
        table.add(&key, &unit, cf, cl, work);
    }
    eprintln!(
        "layout-calibrate: {paired} paired nodes, {changed} changed, channel-first {:.2} ms, channel-last {:.2} ms",
        cf_secs.values().sum::<f64>() * 1e3,
        cl_secs.values().sum::<f64>() * 1e3
    );
    if changed == 0 {
        eprintln!(
            "layout-calibrate: no node differs between the two plans; this host may not flip depthwise convolutions"
        );
    }
    std::fs::write(table_path, table.render())?;
    Ok(())
}

/// `params` is the model as declutter leaves it (every depthwise convolution channel-last); the channel-first and
/// planned plans are rebuilt from `matches` under `TRACT_DISABLE_DEPTHWISE_CHANNEL_LAST` and `TRACT_LAYOUT_PLANNER`.
/// The planned plan reads its table from `TRACT_LAYOUT_TABLE` or the built-in one, not from `--table`.
pub fn handle_predict(
    params: &Parameters,
    matches: &clap::ArgMatches,
    sub: &clap::ArgMatches,
    probe: Option<&Probe>,
) -> TractResult<()> {
    let table_path = sub.get_one::<String>("table").unwrap();
    let table = LayoutTable::parse(&std::fs::read_to_string(table_path)?)
        .with_context(|| format!("Reading table {table_path}"))?;
    let cf_params = rebuild(matches, probe, "TRACT_DISABLE_DEPTHWISE_CHANNEL_LAST")?;
    let est = estimate(&table, typed(&cf_params)?, typed(params)?)?;
    let pct = |d: f64| if est.total > 0.0 { 100.0 * d / est.total } else { 0.0 };
    println!(
        "layout-predict: {} of {} nodes change, all channel-last estimated {:+.1}%",
        est.region_nodes,
        typed(&cf_params)?.nodes().len(),
        pct(est.delta)
    );
    for (key, d) in est.by_key.iter().take(6) {
        println!("  {key:50} {:+6.1}%", pct(*d));
    }
    if est.unknown > 0 {
        println!("  {} nodes with work have no table entry and count for nothing", est.unknown);
    }
    if sub.get_flag("measure") {
        let planned = rebuild(matches, probe, "TRACT_LAYOUT_PLANNER")?;
        let cf = profile_nodes(&cf_params, sub)?.values().sum::<f64>();
        let cl = profile_nodes(params, sub)?.values().sum::<f64>();
        let pl = profile_nodes(&planned, sub)?.values().sum::<f64>();
        let rel = |t: f64| 100.0 * (t / cf - 1.0);
        println!(
            "layout-predict: measured channel-first {:.2} ms, all channel-last {:+.1}%, planned {:+.1}%",
            cf * 1e3,
            rel(cl),
            rel(pl)
        );
    }
    Ok(())
}

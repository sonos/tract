//! `tract <model> -O layout-calibrate --table <file>`: per-op layout costs of the channel-last depthwise flip, measured
//! in the model.
//!
//! The model is built twice, once as declutter leaves it and once with `TRACT_DISABLE_DEPTHWISE_CHANNEL_LAST`, and both
//! optimized plans are profiled. Nodes present under the same name in both plans are paired and grouped by an op key
//! taken from the channel-first plan (op type, plus kernel, stride and channel bucket for depthwise convolutions, output
//! channel bucket and 1x1 / im2col for matrix products and their packing). For each key the table accumulates the
//! channel-first time, the channel-last time and the channel-first work (multiply-adds, or elements produced for ops
//! without arithmetic). An existing table file is added to, so running the command over a set of models builds one
//! table: the ratio of the two times is a key's layout cost, time over work its rate.
//!
//! The two plans only differ on hosts where declutter flips depthwise convolutions, i.e. that have a channel-last
//! depthwise kernel.

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;
use tract_core::internal::*;
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
        .arg(clap::Arg::new("table").long("table").num_args(1).required(true).help("Table file (JSON), added to if it exists"))
}

#[derive(Serialize, Deserialize, Default)]
struct Table {
    models: Vec<String>,
    entries: BTreeMap<String, Entry>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct Entry {
    nodes: usize,
    channel_first_secs: f64,
    channel_last_secs: f64,
    work_unit: String,
    channel_first_work: f64,
}

struct Measured {
    secs: HashMap<String, f64>,
    keys: HashMap<String, (String, String, f64)>,
    shapes: HashMap<String, String>,
}

fn bucket(v: usize, edges: &[usize]) -> usize {
    edges.iter().copied().find(|e| v <= *e).unwrap_or(edges[edges.len() - 1] * 2)
}

fn volume(fact: &TypedFact) -> f64 {
    fact.shape.as_concrete().map(|s| s.iter().product::<usize>() as f64).unwrap_or(0.0)
}

fn fed_by_im2col(model: &TypedModel, node: &TypedNode) -> bool {
    node.inputs.iter().any(|i| model.node(i.node).op.name().to_lowercase().contains("im2col"))
}

fn matmul_key(model: &TypedModel, node: &TypedNode) -> String {
    let Some(mm) = node.op_as::<tract_core::ops::matmul::optimized::OptMatMul>() else {
        return "matmul other".into();
    };
    let shape = mm.c_fact.shape.as_concrete();
    let axis_len = |a: Option<usize>| a.and_then(|a| shape.map(|s| s[a])).unwrap_or(1);
    let (m, n) = (axis_len(mm.c_m_axis), axis_len(mm.c_n_axis));
    if m.min(n) < 4 || m.max(n) < 64 {
        return "matmul gemv".into();
    }
    let kind = if fed_by_im2col(model, node) { "im2col" } else { "1x1" };
    format!("matmul {kind} cout<={}", bucket(m.min(n), &[16, 32, 64, 128, 256]))
}

fn depthwise_key(model: &TypedModel, node: &TypedNode) -> TractResult<String> {
    let input = model.outlet_fact(node.inputs[0])?;
    let kernel = model.outlet_fact(node.inputs[1])?;
    let output = &node.outputs[0].fact;
    let (Some(i), Some(o)) = (input.shape.as_concrete(), output.shape.as_concrete()) else {
        return Ok("depthwise".into());
    };
    let c_axis = (i.len() == 4) as usize;
    let c = i[c_axis];
    let taps = (volume(kernel) as usize) / c.max(1);
    let stride = i[c_axis + 1] / o[c_axis + 1].max(1);
    Ok(format!("depthwise taps{taps} s{stride} c<={}", bucket(c, &[16, 32, 64, 128, 256, 512])))
}

/// The op key, work unit and work of each node of a channel-first plan.
fn keys(model: &TypedModel) -> TractResult<HashMap<String, (String, String, f64)>> {
    let mut out = HashMap::new();
    for node in model.nodes() {
        let inputs = model.node_input_facts(node.id)?;
        let fma: f64 = node
            .op
            .cost(&inputs)?
            .iter()
            .filter(|(c, _)| matches!(c, Cost::FMA(_)))
            .filter_map(|(_, d)| d.to_usize().ok())
            .map(|v| v as f64)
            .sum();
        let name = node.op.name();
        let key = if name == "DepthWiseConv" {
            depthwise_key(model, node)?
        } else if name == "OptMatMul" {
            matmul_key(model, node)
        } else if name == "OptMatMulPack" || name.to_lowercase().contains("im2col") {
            let consumer = node.outputs[0].successors.first().map(|s| model.node(s.node));
            match consumer {
                Some(c) if c.op.name() == "OptMatMul" => {
                    format!("{name} for {}", matmul_key(model, c))
                }
                _ => name.to_string(),
            }
        } else {
            name.to_string()
        };
        let (unit, work) = if fma > 0.0 {
            ("fma", fma)
        } else if name == "OptMatMulPack" || name.to_lowercase().contains("im2col") {
            ("elements", inputs.iter().map(|f| volume(f)).sum())
        } else {
            ("elements", node.outputs.iter().map(|o| volume(&o.fact)).sum())
        };
        out.insert(node.name.clone(), (key, unit.to_string(), work));
    }
    Ok(out)
}

fn measure(params: &Parameters, sub: &clap::ArgMatches) -> TractResult<Measured> {
    let model = params
        .tract_model
        .downcast_ref::<TypedModel>()
        .context("layout-calibrate needs a typed model (pass -O)")?;
    let runnable =
        params.req_runnable().context("layout-calibrate needs an optimized model (pass -O)")?;
    let run_params = run_params_from_subcommand(params, sub)?;
    let inputs = get_or_make_inputs(&params.tract_model, &run_params)?;
    let mut annotations = Annotations::from_model(&*params.tract_model)?;
    profile(&runnable, &bench_limits_from_clap(sub)?, &mut annotations, &inputs, None, false)?;
    let mut secs = HashMap::new();
    for node in model.nodes() {
        if let Some(d) = annotations.tags.get(&node.id.into()).and_then(|t| t.profile) {
            secs.insert(node.name.clone(), d.as_secs_f64());
        }
    }
    let shapes = model
        .nodes()
        .iter()
        .map(|n| {
            (
                n.name.clone(),
                format!("{} {:?}", n.op.name(), n.outputs.first().map(|o| &o.fact.shape)),
            )
        })
        .collect();
    Ok(Measured { secs, keys: keys(model)?, shapes })
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
    let channel_last = measure(params, sub)?;
    let scope =
        tract_core::knobs::KnobScope::new([("TRACT_DISABLE_DEPTHWISE_CHANNEL_LAST", "true")])?;
    let channel_first = scope.enter(|| measure(&Parameters::from_clap(matches, probe)?, sub))?;
    let mut table: Table = match std::fs::read_to_string(table_path) {
        Ok(s) => serde_json::from_str(&s).with_context(|| format!("Reading table {table_path}"))?,
        Err(_) => Table::default(),
    };
    let model = matches.get_one::<String>("model").cloned().unwrap_or_default();
    table.models.push(model);
    let (mut paired, mut changed) = (0, 0);
    for (name, &cf) in &channel_first.secs {
        let (Some(&cl), Some((key, unit, work))) =
            (channel_last.secs.get(name), channel_first.keys.get(name))
        else {
            continue;
        };
        paired += 1;
        if channel_last.shapes.get(name) != channel_first.shapes.get(name) {
            changed += 1;
        }
        let entry = table.entries.entry(key.clone()).or_default();
        entry.nodes += 1;
        entry.channel_first_secs += cf;
        entry.channel_last_secs += cl;
        entry.work_unit = unit.clone();
        entry.channel_first_work += work;
    }
    let total = |m: &Measured| m.secs.values().sum::<f64>() * 1e3;
    eprintln!(
        "layout-calibrate: {paired} paired nodes, {changed} changed, channel-first {:.2} ms, channel-last {:.2} ms",
        total(&channel_first),
        total(&channel_last)
    );
    if changed == 0 {
        eprintln!(
            "layout-calibrate: no node differs between the two plans; this host may not flip depthwise convolutions"
        );
    }
    std::fs::write(table_path, serde_json::to_string_pretty(&table)?)?;
    Ok(())
}

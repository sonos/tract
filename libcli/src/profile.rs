use crate::model::Model;
use crate::tensor::RunTensors;
use crate::tensor::make_inputs_for_model;
use crate::{annotations::*, capture_gpu_trace};
use std::any::TypeId;
use std::time::{Duration, Instant};
use tract_core::internal::*;
use tract_core::num_traits::Zero;
use tract_core::ops::submodel::TypedModelOpState;

pub fn reusable_state(runnable: &Arc<dyn Runnable>) -> bool {
    runnable.typed_model().is_some_and(|model| model.properties().contains_key("pulse.delay"))
}

pub fn run_one_step(
    runnable: &Arc<dyn Runnable>,
    state: &mut Box<dyn State>,
    inputs: &RunTensors,
) -> TractResult<Duration> {
    if !reusable_state(runnable) {
        *state = runnable.spawn()?;
    }
    let start = Instant::now();
    for source in &inputs.sources {
        state.run(source.clone())?;
    }
    Ok(start.elapsed())
}

pub struct BenchLimits {
    pub warmup_loops: usize,
    pub warmup_time: std::time::Duration,
    pub max_loops: usize,
    pub max_time: std::time::Duration,
}

impl Default for BenchLimits {
    fn default() -> Self {
        BenchLimits {
            warmup_loops: 0,
            warmup_time: Duration::default(),
            max_loops: 100_000,
            max_time: std::time::Duration::from_secs(5),
        }
    }
}

/// Structured output of a single bench run: named metrics (e.g. ("evaltime", secs),
/// ("pp512", tok/s)) plus the loop iteration count for the human report line. The
/// `bench`/`llm-bench` runners return this so callers — the interactive subcommand or
/// the bench suite — consume data instead of parsing stdout.
#[derive(Clone, Debug, Default)]
pub struct BenchResult {
    pub metrics: Vec<(String, f64)>,
    pub iters: usize,
}

impl BenchResult {
    /// Emit each metric as a `{"metric":<name>,"value":<f64>}` JSON line on stdout.
    /// This is the bench-suite child→orchestrator contract: stdout is pure JSONL
    /// (logs go to stderr), so the orchestrator can validate every line and treat
    /// anything that does not parse as a hard failure.
    pub fn emit_jsonl(&self) {
        for (k, v) in &self.metrics {
            println!(r#"{{"metric":{k:?},"value":{v}}}"#);
        }
    }
}

/// Load-pipeline checkpoints whose readings the bench suite tracks: the dotted
/// pattern matched against a normalized event label, and the metric-name fragment.
/// The probe writes spaces and dashes as underscores, so `model.ready` matches the
/// `model_ready` line and `before.optimize` matches `after_"before-optimize"`.
const READINGS_STAGES: &[(&str, &str)] =
    &[("model.ready", "model_ready"), ("before.optimize", "before_optimize")];

/// Extract the load-time readings the bench suite reports from a readings-probe
/// output file. For each tracked checkpoint, emit `time_to_<stage>` (elapsed
/// seconds), `rsz_at_<stage>` (resident bytes) and `active_at_<stage>` (alloc −
/// free bytes). A missing file or absent checkpoint is skipped; the orchestrator
/// decides which metrics are required.
pub fn stage_metrics_from_readings(path: impl AsRef<std::path::Path>) -> Vec<(String, f64)> {
    let Ok(content) = std::fs::read_to_string(path) else { return vec![] };
    let normalize = |l: &str| l.replace(['_', '-'], ".");
    let mut out = vec![];
    for (pattern, name) in READINGS_STAGES {
        let Some(line) = content.lines().find(|l| normalize(l).contains(pattern)) else { continue };
        let f: Vec<&str> = line.split_whitespace().collect();
        let parse = |i: usize| f.get(i).and_then(|s| s.parse::<f64>().ok());
        if let (Some(time), Some(rsz), Some(alloc), Some(free)) =
            (parse(0), parse(3), parse(9), parse(10))
        {
            out.push((format!("time_to_{name}"), time));
            out.push((format!("rsz_at_{name}"), rsz));
            out.push((format!("active_at_{name}"), alloc - free));
        }
    }
    out
}

impl BenchLimits {
    pub fn warmup(&self, runnable: &Arc<dyn Runnable>, inputs: &RunTensors) -> TractResult<()> {
        if self.warmup_time.is_zero() && self.warmup_loops.is_zero() {
            return Ok(());
        }
        let reuse = reusable_state(runnable);
        let mut state = runnable.spawn()?;

        let mut iters = 0;
        let max_loops = if self.warmup_loops.is_zero() { usize::MAX } else { self.warmup_loops };
        let max_time = if self.warmup_time.is_zero() { Duration::MAX } else { self.warmup_time };

        let start_warmup = Instant::now();
        info!("Warming up before profiling...");
        while iters < max_loops && start_warmup.elapsed() < max_time {
            if !reuse {
                state = runnable.spawn()?;
            }
            state.run(inputs.sources[0].clone())?;
            iters += 1;
        }
        info!("Done warming up.");

        Ok(())
    }

    pub fn bench(
        &self,
        runnable: &Arc<dyn Runnable>,
        inputs: &RunTensors,
    ) -> TractResult<(usize, Duration)> {
        if self.max_time.is_zero() && self.max_loops.is_zero() {
            return Ok(Default::default());
        }
        let reuse = reusable_state(runnable);
        let mut state = runnable.spawn()?;

        let mut iters = 0;
        let max_loops = if self.max_loops.is_zero() { usize::MAX } else { self.max_loops };
        let max_time = if self.max_time.is_zero() { Duration::MAX } else { self.max_time };

        let mut dur = Duration::default();
        let start = Instant::now();
        while iters < max_loops && start.elapsed() < max_time {
            if !reuse {
                state = runnable.spawn()?;
            }
            let start_inner = Instant::now();
            state.run(inputs.sources[0].clone())?;
            dur += start_inner.elapsed();
            iters += 1;
        }

        Ok((iters, dur))
    }
}

pub fn profile(
    runnable: &Arc<dyn Runnable>,
    bench_limits: &BenchLimits,
    dg: &mut Annotations,
    inputs: &RunTensors,
    custom_profiler: Option<HashMap<TypeId, Profiler>>,
    folded: bool,
) -> TractResult<()> {
    let Some(plan) = runnable.typed_plan() else {
        bail!("Can only profile TypedRunnable");
    };
    info!("Running entire network");
    let mut iters = 0usize;
    let prefix = tvec!();

    bench_limits.warmup(runnable, inputs)?;

    let reuse = reusable_state(runnable);
    let mut state = plan.spawn()?;

    let mut dur = Duration::default();
    let mut time_accounted_by_inner_nodes = Duration::default();
    while iters < bench_limits.max_loops && dur < bench_limits.max_time {
        if !reuse {
            state = plan.spawn()?;
        }
        let start = Instant::now();

        for source in &inputs.sources {
            rec_profiler(
                &mut state,
                dg,
                source,
                custom_profiler.as_ref(),
                &prefix,
                None,
                &mut time_accounted_by_inner_nodes,
                folded,
            )?;
        }
        dur += start.elapsed();
        iters += 1;
    }

    dur -= time_accounted_by_inner_nodes;

    info!("Running {} iterations max. for each node.", bench_limits.max_loops);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

    let denum = (iters as f32).recip();
    let entire = dur.mul_f32(denum);
    for d in dg.tags.values_mut() {
        if let Some(d) = d.profile.as_mut() {
            *d = d.mul_f32(denum);
        }

        if let Some(d) = d.accelerator_profile.as_mut() {
            *d = d.mul_f32(denum);
        }
    }
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    let accel_sum = dg.tags.values().filter_map(|t| t.accelerator_profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, accel_sum, entire, iters });
    Ok(())
}

#[allow(clippy::type_complexity)]
pub fn profile_gpu(
    runnable: &Arc<dyn Runnable>,
    bench_limits: &BenchLimits,
    sub_matches: &clap::ArgMatches,
    dg: &mut Annotations,
    inputs: &RunTensors,
    before_node: &dyn Fn(usize),
    after_iteration: &dyn Fn(&mut Annotations, &[(usize, String)]) -> TractResult<()>,
) -> TractResult<()> {
    let Some(plan) = runnable.typed_plan() else {
        bail!("Can only profile TypedRunnable");
    };
    info!("Running entire network");
    let mut iters = 0usize;
    let prefix = tvec!();

    bench_limits.warmup(runnable, inputs)?;

    let reuse = reusable_state(runnable);
    let mut state = plan.spawn()?;

    let mut dur = Duration::default();

    capture_gpu_trace(sub_matches, || -> TractResult<()> {
        while iters < bench_limits.max_loops && dur < bench_limits.max_time {
            if !reuse {
                state = plan.spawn()?;
            }
            let start = Instant::now();
            for source in &inputs.sources {
                rec_profiler_gpu(&mut state, dg, source, &prefix, before_node)?;
            }
            after_iteration(dg, &prefix)?;
            dur += start.elapsed();
            iters += 1;
        }
        Ok(())
    })?;

    info!("Running {} iterations max. for each node.", bench_limits.max_loops);
    info!("Running for {} ms max. for each node.", bench_limits.max_time.as_millis());

    let denum = (iters as f32).recip();
    let entire = dur.mul_f32(denum);
    for d in dg.tags.values_mut() {
        if let Some(d) = d.profile.as_mut() {
            *d = d.mul_f32(denum);
        }

        if let Some(d) = d.accelerator_profile.as_mut() {
            *d = d.mul_f32(denum);
        }
    }
    let max = dg.tags.values().filter_map(|t| t.profile).max().unwrap();
    let sum = dg.tags.values().filter_map(|t| t.profile).sum::<Duration>();
    let accel_sum = dg.tags.values().filter_map(|t| t.accelerator_profile).sum::<Duration>();
    dg.profile_summary = Some(ProfileSummary { max, sum, accel_sum, entire, iters });
    Ok(())
}

pub fn rec_profiler_gpu(
    state: &mut TypedSimpleState,
    dg: &mut Annotations,
    inputs: &TVec<TValue>,
    prefix: &[(usize, String)],
    before_node: &dyn Fn(usize),
) -> TractResult<TVec<TValue>> {
    let r = state.run_plan_with_eval(
        inputs.clone(),
        |session_state, mut node_state, node, input| {
            before_node(node.id);
            // Profile node
            let start = crate::time::now();
            let res = tract_core::plan::eval(
                session_state,
                node_state.as_deref_mut(),
                node,
                input.clone(),
            );
            let elapsed = start.elapsed();
            let node_id = NodeQId(prefix.into(), node.id);
            *dg.node_mut(node_id).profile.get_or_insert(Duration::default()) += elapsed;

            res
        },
    )?;

    Ok(r)
}

#[allow(clippy::too_many_arguments)]
pub fn rec_profiler(
    state: &mut TypedSimpleState,
    dg: &mut Annotations,
    inputs: &TVec<TValue>,
    profilers: Option<&HashMap<TypeId, Profiler>>,
    prefix: &[(usize, String)],
    multiplier: Option<usize>,
    time_accounted_by_inner_nodes: &mut Duration,
    folded: bool,
) -> TractResult<TVec<TValue>> {
    let r = state.run_plan_with_eval(
        inputs.clone(),
        |session_state, mut node_state, node, input| {
            // Profile node
            let start = crate::time::now();
            let res = tract_core::plan::eval(
                session_state,
                node_state.as_deref_mut(),
                node,
                input.clone(),
            );
            let elapsed = start.elapsed().mul_f32(multiplier.unwrap_or(1) as _);
            let node_id = NodeQId(prefix.into(), node.id);
            *dg.node_mut(node_id).profile.get_or_insert(Duration::default()) += elapsed;

            if !folded {
                let start = crate::time::now();
                profile_submodel(
                    node,
                    node_state,
                    input,
                    dg,
                    profilers,
                    prefix,
                    time_accounted_by_inner_nodes,
                )?;
                *time_accounted_by_inner_nodes += start.elapsed();
            }

            // Update parent nodes if any (childs timings are deducted from parents)
            let prefix_vec = prefix.to_vec();
            if !prefix_vec.is_empty() {
                (1..prefix_vec.len() + 1).map(|idx| prefix_vec[..idx].to_vec()).for_each(
                    |parent_path| {
                        let parent_node = parent_path.last().map(|it| it.0).unwrap();
                        let parent = dg
                            .node_mut(NodeQId(
                                parent_path[..parent_path.len() - 1].into(),
                                parent_node,
                            ))
                            .profile
                            .get_or_insert(Duration::default());
                        *parent -= elapsed.min(*parent);
                    },
                );
            }
            res
        },
    )?;
    Ok(r)
}

fn profile_submodel(
    node: &TypedNode,
    mut node_state: Option<&mut dyn OpState>,
    input: TVec<TValue>,
    dg: &mut Annotations,
    profilers: Option<&HashMap<TypeId, Profiler>>,
    prefix: &[(usize, String)],
    time_accounted_by_inner_nodes: &mut Duration,
) -> TractResult<()> {
    if let Some(ref mut op_state) = node_state {
        if let Some(profiler) = profilers.and_then(|it| it.get(&op_state.type_id())) {
            let mut new_prefix: TVec<_> = prefix.into();
            new_prefix.push((node.id, "submodel".to_string()));

            let (_, _) =
                (profiler.func)(*op_state, input, dg, &new_prefix, time_accounted_by_inner_nodes)?;
        } else if let Some(scan_state) = op_state.downcast_mut::<tract_core::ops::scan::State>() {
            let mut new_prefix: TVec<_> = prefix.into();
            new_prefix.push((node.id, "loop".to_string()));

            let scan_inputs = make_inputs_for_model(scan_state.model_state.model())?;
            let multi = scan_state.iteration_count(&input);

            rec_profiler(
                &mut scan_state.model_state,
                dg,
                &scan_inputs,
                None,
                &new_prefix,
                Some(multi),
                time_accounted_by_inner_nodes,
                false,
            )?;
        } else if let Some(typed_model_state) = op_state.downcast_mut::<TypedModelOpState>() {
            let mut new_prefix: TVec<_> = prefix.into();
            new_prefix.push((node.id, "submodel".to_string()));

            rec_profiler(
                typed_model_state,
                dg,
                &input,
                None,
                &new_prefix,
                None,
                time_accounted_by_inner_nodes,
                false,
            )?;
        }
    }

    Ok(())
}

type ProfilerFn = fn(
    &mut dyn OpState,
    TVec<TValue>,
    &mut Annotations,
    &[(usize, String)],
    &mut Duration,
) -> TractResult<(TractResult<TVec<TValue>>, Duration)>;

#[derive(Clone)]
pub struct Profiler {
    pub func: ProfilerFn,
    pub name: &'static str,
}

impl Hash for Profiler {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

pub fn extract_costs(
    annotations: &mut Annotations,
    model: &dyn Model,
    extra_symbols: &SymbolValues,
) -> TractResult<()> {
    fn extract_costs_rec(
        annotations: &mut Annotations,
        model: &dyn Model,
        prefix: &[(usize, String)],
        multiplier: TDim,
        extra_symbols: &SymbolValues,
    ) -> TractResult<()> {
        if let Some(model) = model.downcast_ref::<TypedModel>() {
            for node_id in 0..model.nodes().len() {
                let inputs = model.node_input_facts(node_id)?;
                let cost = model
                    .node(node_id)
                    .op
                    .cost(&inputs)
                    .with_context(|| format!("costing node {}", model.node(node_id)))?;
                annotations.node_mut(NodeQId(prefix.into(), node_id)).cost = cost
                    .into_iter()
                    .map(|(k, v)| {
                        let cost = if k.is_compute() { v * &multiplier } else { v };
                        (k, cost.eval(extra_symbols))
                    })
                    .collect();

                let nested_subs = model.nested_models(node_id);
                let nested_multis = (model as &dyn Model).nested_models_iters(node_id, &inputs);
                for (name, sub) in nested_subs {
                    let mut prefix: TVec<_> = prefix.into();
                    prefix.push((node_id, name.to_string()));
                    extract_costs_rec(
                        annotations,
                        sub,
                        &prefix,
                        nested_multis.clone().unwrap_or_else(|| 1.into()) * &multiplier,
                        extra_symbols,
                    )?;
                }
            }
        }
        Ok(())
    }
    extract_costs_rec(annotations, model, &[], 1.into(), extra_symbols)
}

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

    /// One saturating loop per stream: `streams` threads, a state each, every one
    /// of them feeding turns as fast as it can until the limits run out. The
    /// limits are per stream, the wall clock is shared.
    ///
    /// This is what a laned runtime is compared against: the same load offered
    /// to the plain runtime is `streams` independent states running side by
    /// side.
    // Wasm has no threads to saturate with.
    #[cfg(not(target_family = "wasm"))]
    pub fn bench_streams(
        &self,
        runnable: &Arc<dyn Runnable>,
        inputs: &RunTensors,
        streams: usize,
    ) -> TractResult<StreamsBench> {
        ensure!(streams > 0, "A saturating bench needs a stream at least");
        let reuse = reusable_state(runnable);
        let max_loops = if self.max_loops.is_zero() { usize::MAX } else { self.max_loops };
        let max_time = if self.max_time.is_zero() { Duration::MAX } else { self.max_time };
        let start = Instant::now();
        let turns: Vec<Vec<Duration>> = std::thread::scope(|scope| {
            let running: Vec<_> = (0..streams)
                .map(|_| {
                    scope.spawn(|| -> TractResult<Vec<Duration>> {
                        let mut state = runnable.spawn()?;
                        let mut turns = vec![];
                        while turns.len() < max_loops && start.elapsed() < max_time {
                            if !reuse {
                                state = runnable.spawn()?;
                            }
                            let turn = Instant::now();
                            state.run(inputs.sources[0].clone())?;
                            turns.push(turn.elapsed());
                        }
                        Ok(turns)
                    })
                })
                .collect();
            running.into_iter().map(|stream| stream.join().unwrap()).collect::<Vec<_>>()
        })
        .into_iter()
        .collect::<TractResult<_>>()?;
        let wall = start.elapsed();
        let mut latencies: Vec<Duration> = turns.into_iter().flatten().collect();
        latencies.sort();
        Ok(StreamsBench { streams, wall, latencies })
    }

    /// One paced loop per stream: `streams` threads, a state each, every one of
    /// them owing a turn every period of the wall clock, the way a live stream
    /// does. Turn `n` of a stream is due at a fixed offset from its first, so a
    /// turn which runs long does not push the ones behind it back -- the stream
    /// catches up or falls further behind, and `added` is what says which.
    ///
    /// Streams are phased evenly over one period. Each session runs one turn
    /// before the clock starts, so the buffers a first turn allocates are not
    /// charged to the load.
    ///
    /// Under `churn` a stream is a seat in a steady population rather than one
    /// session: it holds a session for a draw around `hold`, gives it up, and
    /// admits another in its place, so the lanes turn over while the load does
    /// not.
    #[cfg(not(target_family = "wasm"))]
    pub fn bench_streams_paced(
        &self,
        runnable: &Arc<dyn Runnable>,
        inputs: &RunTensors,
        streams: usize,
        pacing: &Pacing,
    ) -> TractResult<PacedBench> {
        use rand::SeedableRng;
        ensure!(streams > 0, "A paced bench needs a stream at least");
        ensure!(!pacing.period.is_zero(), "A paced bench needs a turn period");
        let max_loops = if self.max_loops.is_zero() { usize::MAX } else { self.max_loops };
        let max_time = if self.max_time.is_zero() { Duration::MAX } else { self.max_time };
        ensure!(
            max_time >= 4 * pacing.period,
            "A turn period of {:?} wants four periods of run at least, --max-time gives {max_time:?}",
            pacing.period
        );
        let reuse = reusable_state(runnable);
        ensure!(
            pacing.churn.is_none() || reuse,
            "Churn ends a session and admits another in its place, which only means something for a model carrying a session across turns"
        );
        let start = Instant::now();
        let served: Vec<StreamTurns> = std::thread::scope(|scope| {
            let running: Vec<_> = (0..streams)
                .map(|stream| {
                    scope.spawn(move || -> TractResult<StreamTurns> {
                        let mut rng = rand::rngs::StdRng::seed_from_u64(stream as u64);
                        let phase = pacing.period.mul_f64(stream as f64 / streams as f64);
                        let mut turns = StreamTurns::default();
                        // The seat's slots in the period, one turn owed at each.
                        // Laid down once the first session has run its pre-clock
                        // turn -- 144 of those serialize through the worker, and
                        // a grid laid before them starts every stream already
                        // behind -- and never re-phased, so churn costs what
                        // admitting a session costs rather than what bunching
                        // every arrival would.
                        let mut grid: Option<Instant> = None;
                        let mut slot = 0usize;
                        while turns.service.len() < max_loops && start.elapsed() < max_time {
                            let (mut state, starved, took) = admit(runnable, &start, max_time)?;
                            turns.admissions += 1;
                            turns.starved += usize::from(starved);
                            turns.admitting.push(took);
                            // A session's first turn allocates its buffers, and
                            // runs before its clock starts, so the load is not
                            // charged for it.
                            state.run(inputs.sources[0].clone())?;
                            let admitted = Instant::now();
                            let grid = *grid.get_or_insert(admitted + phase);
                            // A session serves the seat's slots which fall in
                            // its own life, starting at the first one at or
                            // after it arrived: a stream's clock starts when it
                            // does, so a session does not inherit what the seat
                            // was owed before it.
                            if let Some(late) = admitted.checked_duration_since(grid) {
                                let missed = late.div_duration_f64(pacing.period).ceil();
                                slot = slot.max(missed as usize);
                            }
                            let hold = pacing
                                .churn
                                .map(|churn| exponential(&mut rng, churn.hold).max(pacing.period));
                            while turns.service.len() < max_loops && start.elapsed() < max_time {
                                if hold.is_some_and(|hold| admitted.elapsed() >= hold) {
                                    break;
                                }
                                if !reuse {
                                    state = runnable.spawn()?;
                                }
                                let due = grid + pacing.period * slot as u32;
                                slot += 1;
                                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                                    std::thread::sleep(wait);
                                }
                                let turn = Instant::now();
                                state.run(inputs.sources[0].clone())?;
                                let done = Instant::now();
                                turns.service.push(done - turn);
                                turns.added.push(done.saturating_duration_since(due));
                                turns.span = done - grid;
                            }
                        }
                        Ok(turns)
                    })
                })
                .collect();
            running.into_iter().map(|stream| stream.join().unwrap()).collect::<Vec<_>>()
        })
        .into_iter()
        .collect::<TractResult<_>>()?;
        let wall = served.iter().map(|turns| turns.span).max().unwrap_or_default();
        let admissions = served.iter().map(|turns| turns.admissions).sum();
        let starved = served.iter().map(|turns| turns.starved).sum();
        let mut admitting: Vec<Duration> =
            served.iter().flat_map(|turns| turns.admitting.iter().copied()).collect();
        admitting.sort();
        let mut service: Vec<Duration> =
            served.iter().flat_map(|turns| turns.service.iter().copied()).collect();
        let mut added: Vec<Duration> = served.into_iter().flat_map(|turns| turns.added).collect();
        service.sort();
        added.sort();
        Ok(PacedBench {
            streams,
            deadline: pacing.deadline,
            wall,
            service,
            added,
            admissions,
            starved,
            admitting,
        })
    }
}

/// Take a session, retrying while every lane is held. A laned runtime hands a
/// lane back through its worker's queue, so a session leaving and another
/// arriving in its place can find none free for a moment: that is the starve
/// count, and how long it waited shows up as the new session's first turns
/// being late.
#[cfg(not(target_family = "wasm"))]
fn admit(
    runnable: &Arc<dyn Runnable>,
    start: &Instant,
    max_time: Duration,
) -> TractResult<(Box<dyn State>, bool, Duration)> {
    let mut starved = false;
    let asked = Instant::now();
    loop {
        match runnable.spawn() {
            Ok(state) => return Ok((state, starved, asked.elapsed())),
            Err(e) => {
                if start.elapsed() >= max_time {
                    return Err(e).context("Admitting a session");
                }
                starved = true;
                std::thread::sleep(Duration::from_micros(200));
            }
        }
    }
}

/// A holding time around `mean`, so sessions do not all leave together.
#[cfg(not(target_family = "wasm"))]
fn exponential(rng: &mut impl rand::RngExt, mean: Duration) -> Duration {
    let uniform: f64 = rng.random::<f64>();
    mean.mul_f64(-(1. - uniform).max(f64::MIN_POSITIVE).ln())
}

/// The wall clock a paced load runs against: what a turn's input covers, and
/// how much a turn may add over its arrival before it counts as late.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub struct Pacing {
    pub period: Duration,
    pub deadline: Duration,
    pub churn: Option<Churn>,
}

/// Sessions coming and going under a steady population: how long one is held
/// before it leaves and another is admitted in its place.
#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy, Debug)]
pub struct Churn {
    pub hold: Duration,
}

/// One stream's turns, as it served them.
#[cfg(not(target_family = "wasm"))]
#[derive(Default)]
struct StreamTurns {
    service: Vec<Duration>,
    added: Vec<Duration>,
    span: Duration,
    admissions: usize,
    starved: usize,
    admitting: Vec<Duration>,
}

/// What `streams` paced streams did against that clock: how long each turn took
/// to run, and how late each one completed against the moment its input
/// arrived. The second is what a real-time load is judged on -- it holds the
/// queueing a turn waited through as well as its own run.
#[cfg(not(target_family = "wasm"))]
pub struct PacedBench {
    pub streams: usize,
    pub deadline: Duration,
    pub wall: Duration,
    /// One entry per turn, over every stream, sorted.
    pub service: Vec<Duration>,
    /// One entry per turn, over every stream, sorted.
    pub added: Vec<Duration>,
    /// Sessions admitted over the trial: one per stream without churn, and
    /// however many the holding time turned over with it.
    pub admissions: usize,
    /// Admissions which found every lane held and had to wait for one.
    pub starved: usize,
    /// What each admission waited between asking for a session and getting one,
    /// sorted. A laned runtime answers on its worker, between turns, so this
    /// holds the turn in flight as well as the lane's own reset -- it is the
    /// wait, not the cost of the reset. Read it at a quantile: every seat's
    /// first admission lands in one storm at the start of the trial, so a mean
    /// over a short trial is mostly that storm.
    pub admitting: Vec<Duration>,
}

#[cfg(not(target_family = "wasm"))]
impl PacedBench {
    /// Sessions admitted per second: how fast the lanes turn over.
    pub fn admissions_per_second(&self) -> f64 {
        self.admissions as f64 / self.wall.as_secs_f64()
    }

    /// The wait `q` of the admissions came in under, `q` in 0..1.
    pub fn admission_quantile(&self, q: f64) -> Duration {
        quantile(&self.admitting, q)
    }

    /// The share of admissions which waited for a lane.
    pub fn starve_fraction(&self) -> f64 {
        if self.admissions == 0 {
            return 0.;
        }
        self.starved as f64 / self.admissions as f64
    }

    pub fn turns(&self) -> usize {
        self.service.len()
    }

    pub fn turns_per_second(&self) -> f64 {
        self.turns() as f64 / self.wall.as_secs_f64()
    }

    pub fn mean_service(&self) -> Duration {
        self.service.iter().sum::<Duration>().checked_div(self.turns() as u32).unwrap_or_default()
    }

    /// The added latency `q` of the turns came in under, `q` in 0..1.
    pub fn added_quantile(&self, q: f64) -> Duration {
        quantile(&self.added, q)
    }

    /// The share of turns which completed past the deadline.
    pub fn late_fraction(&self) -> f64 {
        if self.added.is_empty() {
            return 0.;
        }
        let in_time = self.added.partition_point(|added| *added <= self.deadline);
        (self.turns() - in_time) as f64 / self.turns() as f64
    }

    /// Whether this load held: the `q`th added latency met the deadline. A load
    /// which served nothing never holds.
    pub fn holds(&self, q: f64) -> bool {
        !self.added.is_empty() && self.added_quantile(q) <= self.deadline
    }
}

/// The value `q` of a sorted series comes in under, `q` in 0..1.
#[cfg(not(target_family = "wasm"))]
fn quantile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::default();
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// What `streams` saturating streams did: how long they ran for, and how long
/// every one of their turns took.
#[cfg(not(target_family = "wasm"))]
pub struct StreamsBench {
    pub streams: usize,
    pub wall: Duration,
    /// One entry per turn, over every stream, sorted.
    pub latencies: Vec<Duration>,
}

#[cfg(not(target_family = "wasm"))]
impl StreamsBench {
    pub fn turns(&self) -> usize {
        self.latencies.len()
    }

    /// Turns served per second, all streams together: the number a laned
    /// runtime has to beat.
    pub fn turns_per_second(&self) -> f64 {
        self.turns() as f64 / self.wall.as_secs_f64()
    }

    pub fn mean_latency(&self) -> Duration {
        self.latencies.iter().sum::<Duration>().checked_div(self.turns() as u32).unwrap_or_default()
    }

    /// The latency `q` of the turns came in under, `q` in 0..1.
    pub fn latency_quantile(&self, q: f64) -> Duration {
        quantile(&self.latencies, q)
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
    let r = state.run_plan_with_eval(inputs.clone(), |turn, mut node_state, node, input| {
        before_node(node.id);
        // Profile node
        let start = crate::time::now();
        let res = tract_core::plan::eval(turn, node_state.as_deref_mut(), node, input.clone());
        let elapsed = start.elapsed();
        let node_id = NodeQId(prefix.into(), node.id);
        *dg.node_mut(node_id).profile.get_or_insert(Duration::default()) += elapsed;

        res
    })?;

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
    let r = state.run_plan_with_eval(inputs.clone(), |turn, mut node_state, node, input| {
        // Keep a copy of the inputs only when a nested submodel will need them
        // for recursive profiling. Otherwise move them straight into eval: an
        // extra clone here holds a second Arc ref to each input and forces
        // in-place ops (reshape, by-scalar/unicast bias add, ...) down their
        // copy-on-shared path, inflating their measured time versus production.
        let saved_input = (!folded && node_state.is_some()).then(|| input.clone());
        // Profile node
        let start = crate::time::now();
        let res = tract_core::plan::eval(turn, node_state.as_deref_mut(), node, input);
        let elapsed = start.elapsed().mul_f32(multiplier.unwrap_or(1) as _);
        let node_id = NodeQId(prefix.into(), node.id);
        *dg.node_mut(node_id).profile.get_or_insert(Duration::default()) += elapsed;

        if let Some(saved_input) = saved_input {
            let start = crate::time::now();
            profile_submodel(
                node,
                node_state,
                saved_input,
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
                        .node_mut(NodeQId(parent_path[..parent_path.len() - 1].into(), parent_node))
                        .profile
                        .get_or_insert(Duration::default());
                    *parent -= elapsed.min(*parent);
                },
            );
        }
        res
    })?;
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

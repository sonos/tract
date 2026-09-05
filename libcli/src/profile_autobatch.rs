//! Benching a model which serves several sessions at once: a saturating load,
//! a paced one, and what a capacity search reports of either.
//!
//! Split from [`super::profile`] because none of it works without threads, so
//! the whole module is gated off on Wasm rather than every item in it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tract_core::internal::*;
use tract_core::num_traits::Zero;
use tract_core::runtime::Runnable;

use crate::profile::{BenchLimits, reusable_state};
use crate::tensor::RunTensors;

impl BenchLimits {
    /// One saturating loop per stream: `streams` threads, a state each, every one
    /// of them feeding turns as fast as it can until the limits run out. The
    /// limits are per stream, the wall clock is shared.
    ///
    /// This is what an autobatch runtime is compared against: the same load offered
    /// to the plain runtime is `streams` independent states running side by
    /// side.
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

/// Take a session, retrying while every lane is held. An autobatch runtime hands a
/// lane back through its worker's queue, so a session leaving and another
/// arriving in its place can find none free for a moment: that is the starve
/// count, and how long it waited shows up as the new session's first turns
/// being late.
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
fn exponential(rng: &mut impl rand::RngExt, mean: Duration) -> Duration {
    let uniform: f64 = rng.random::<f64>();
    mean.mul_f64(-(1. - uniform).max(f64::MIN_POSITIVE).ln())
}

/// The wall clock a paced load runs against: what a turn's input covers, and
/// how much a turn may add over its arrival before it counts as late.
#[derive(Clone, Copy, Debug)]
pub struct Pacing {
    pub period: Duration,
    pub deadline: Duration,
    pub churn: Option<Churn>,
}

/// Sessions coming and going under a steady population: how long one is held
/// before it leaves and another is admitted in its place.
#[derive(Clone, Copy, Debug)]
pub struct Churn {
    pub hold: Duration,
}

/// One stream's turns, as it served them.
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
    /// sorted. An autobatch runtime answers on its worker, between turns, so this
    /// holds the turn in flight as well as the lane's own reset -- it is the
    /// wait, not the cost of the reset. Read it at a quantile: every seat's
    /// first admission lands in one storm at the start of the trial, so a mean
    /// over a short trial is mostly that storm.
    pub admitting: Vec<Duration>,
}

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
fn quantile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::default();
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// What `streams` saturating streams did: how long they ran for, and how long
/// every one of their turns took.
pub struct StreamsBench {
    pub streams: usize,
    pub wall: Duration,
    /// One entry per turn, over every stream, sorted.
    pub latencies: Vec<Duration>,
}

impl StreamsBench {
    pub fn turns(&self) -> usize {
        self.latencies.len()
    }

    /// Turns served per second, all streams together: the number an autobatch
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

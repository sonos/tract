//! Serving concurrent callers through a fixed set of states.
//!
//! [`PooledRunnable::wrap`] starts [`TRACT_POOL_WORKERS`] threads, each holding
//! one state of the inner runnable for the runnable's whole life, and feeds them
//! from one bounded queue. Whatever the number of threads calling `run()`, the
//! model runs on that many states only: on a GPU, that many streams, BLAS and
//! DNN handles and arenas, all allocated before `wrap` returns, so a device too
//! small for them fails the load and not a call under traffic.
//!
//! A call goes to whichever worker is free, so consecutive calls of one caller
//! land on different states. That is only sound for a model whose outputs
//! depend on the call's inputs alone: `wrap` refuses a pulsed model, and any
//! other state carried from one call to the next is on trust.
//!
//! A call finding [`TRACT_POOL_QUEUE`] calls already waiting fails at once with
//! [`Overloaded`], which a server can answer with a "retry later" instead of
//! queueing without bound.
//!
//! A call panicking fails with an error carrying the panic message, and the
//! worker replaces its state with a fresh one before taking the next call.
//!
//! Dropping the runnable closes the queue and joins the workers: their
//! per-thread device state must be torn down before the libraries are.

use std::any::Any;
use std::fmt::Debug;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::thread;
use std::time::Instant;

use metrics::{Counter, Gauge, Histogram, Label, Unit};

use crate::internal::*;

crate::declare_knob!(
    scopable TRACT_POOL_WORKERS,
    Option<usize>,
    None,
    "States a deployment config's model is served through, one thread each; unset, it is not pooled."
);

crate::declare_knob!(
    scopable TRACT_POOL_QUEUE,
    usize,
    64,
    "Most calls a pooled model holds waiting for a worker; one more fails as overloaded."
);

/// The error a [`PooledRunnable`] call fails with when its queue is full. Find
/// it with `downcast_ref` on the returned error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overloaded;

impl std::fmt::Display for Overloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Overloaded: every pool worker is busy and the queue is full")
    }
}

impl std::error::Error for Overloaded {}

/// A model served through a fixed number of states, see the module
/// documentation.
#[derive(Clone)]
pub struct PooledRunnable {
    shared: Arc<Shared>,
}

struct Shared {
    /// `None` once the runnable is being dropped, which is what closes the
    /// queue and stops the workers.
    calls: Option<SyncSender<Call>>,
    workers: Vec<thread::JoinHandle<()>>,
    inner: Arc<dyn Runnable>,
    queue: usize,
    meters: Meters,
}

impl Drop for Shared {
    fn drop(&mut self) {
        self.calls.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

struct Call {
    inputs: TVec<TValue>,
    queued: Instant,
    done: Sender<TractResult<TVec<TValue>>>,
}

/// What the pool reports through the `metrics` facade, labelled `runnable`
/// with the model's `tract.name` when it has one.
#[derive(Clone)]
struct Meters {
    queued: Gauge,
    rejected: Counter,
    queue_wait: Histogram,
    run: Histogram,
    panics: Counter,
}

impl Meters {
    fn new(name: Option<&str>) -> Meters {
        metrics::describe_gauge!("tract_pool_queued", "Calls waiting for a pool worker");
        metrics::describe_counter!(
            "tract_pool_rejected_total",
            "Calls a pooled runnable refused on a full queue"
        );
        metrics::describe_histogram!(
            "tract_pool_queue_wait_seconds",
            Unit::Seconds,
            "From a call queued to a pool worker taking it"
        );
        metrics::describe_histogram!(
            "tract_pool_run_seconds",
            Unit::Seconds,
            "One call on a pool worker's state"
        );
        metrics::describe_counter!(
            "tract_pool_worker_panics_total",
            "Calls a pool worker panicked on, each costing it its state"
        );
        let labels: Vec<Label> =
            name.map(|name| Label::new("runnable", name.to_string())).into_iter().collect();
        Meters {
            queued: metrics::gauge!("tract_pool_queued", labels.clone()),
            rejected: metrics::counter!("tract_pool_rejected_total", labels.clone()),
            queue_wait: metrics::histogram!("tract_pool_queue_wait_seconds", labels.clone()),
            run: metrics::histogram!("tract_pool_run_seconds", labels.clone()),
            panics: metrics::counter!("tract_pool_worker_panics_total", labels),
        }
    }
}

impl PooledRunnable {
    /// Serve `inner` through `workers` states, with at most `queue` calls
    /// waiting for one. Every state is spawned before this returns, and the
    /// first spawn failing fails the wrap.
    pub fn wrap(
        inner: Arc<dyn Runnable>,
        workers: usize,
        queue: usize,
    ) -> TractResult<PooledRunnable> {
        ensure!(workers > 0, "A pool needs one worker at least");
        if let Some(key) = inner.properties().keys().find(|key| key.starts_with("pulse.")) {
            bail!(
                "A pulsed model (it carries {key}) keeps its stream between calls, and a pool \
                 hands each call to whichever worker is free"
            );
        }
        let meters = Meters::new(inner.name());
        let (calls, queued) = sync_channel::<Call>(queue);
        let queued = Arc::new(Mutex::new(queued));
        let (spawned, ready) = channel::<TractResult<()>>();
        let mut shared =
            Shared { calls: Some(calls), workers: vec![], inner: inner.clone(), queue, meters };
        for _ in 0..workers {
            let inner = inner.clone();
            let queued = queued.clone();
            let spawned = spawned.clone();
            let meters = shared.meters.clone();
            shared.workers.push(thread::Builder::new().name("tract-pool".into()).spawn(
                move || match inner.spawn() {
                    Ok(state) => {
                        let _ = spawned.send(Ok(()));
                        work(&inner, state, &queued, &meters);
                    }
                    Err(e) => {
                        let _ = spawned.send(Err(e));
                    }
                },
            )?);
        }
        for _ in 0..workers {
            ready
                .recv()
                .map_err(|_| format_err!("A pool worker died spawning its state"))?
                .context("Spawning a pool worker's state")?;
        }
        Ok(PooledRunnable { shared: Arc::new(shared) })
    }

    pub fn workers(&self) -> usize {
        self.shared.workers.len()
    }

    pub fn queue(&self) -> usize {
        self.shared.queue
    }

    pub fn inner(&self) -> &Arc<dyn Runnable> {
        &self.shared.inner
    }

    fn call(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let calls = self.shared.calls.as_ref().context("The pool is gone")?;
        let (done, outputs) = channel();
        let meters = &self.shared.meters;
        meters.queued.increment(1);
        if let Err(e) = calls.try_send(Call { inputs, queued: Instant::now(), done }) {
            meters.queued.decrement(1);
            return match e {
                TrySendError::Full(_) => {
                    meters.rejected.increment(1);
                    Err(Overloaded.into())
                }
                TrySendError::Disconnected(_) => bail!("The pool workers are gone"),
            };
        }
        outputs.recv().map_err(|_| format_err!("A pool worker dropped a call"))?
    }
}

/// One worker's life: take the next call, run it on the worker's own state,
/// answer it, until the queue is closed. A call panicking is answered with an
/// error and costs the worker its state, which is replaced by a fresh one; the
/// worker stops only if that spawn fails.
fn work(
    inner: &Arc<dyn Runnable>,
    mut state: Box<dyn State>,
    queued: &Mutex<Receiver<Call>>,
    meters: &Meters,
) {
    loop {
        let call = match queued.lock() {
            Ok(queued) => queued.recv(),
            Err(_) => return,
        };
        let Ok(call) = call else { return };
        meters.queued.decrement(1);
        meters.queue_wait.record(call.queued.elapsed());
        let start = Instant::now();
        let inputs = call.inputs;
        // The state is dropped after a panic, so nothing observes it half-updated.
        let outputs = catch_unwind(AssertUnwindSafe(|| state.run(inputs)));
        meters.run.record(start.elapsed());
        match outputs {
            Ok(outputs) => {
                let _ = call.done.send(outputs);
            }
            Err(panic) => {
                meters.panics.increment(1);
                let _ = call.done.send(Err(format_err!(
                    "A pool worker panicked running the call: {}",
                    panic_message(&*panic)
                )));
                state = match inner.spawn() {
                    Ok(state) => state,
                    Err(e) => {
                        log::error!("A pool worker could not replace its state, it stops: {e:?}");
                        return;
                    }
                };
            }
        }
    }
}

fn panic_message(panic: &(dyn Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(|s| s.as_str()))
        .unwrap_or("no message")
}

impl Debug for PooledRunnable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "PooledRunnable({} workers, queue {})", self.workers(), self.queue())
    }
}

impl Runnable for PooledRunnable {
    fn run(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.call(inputs)
    }

    fn spawn(&self) -> TractResult<Box<dyn State>> {
        Ok(Box::new(PooledStateHandle { runnable: self.clone() }))
    }

    fn input_count(&self) -> usize {
        self.shared.inner.input_count()
    }

    fn output_count(&self) -> usize {
        self.shared.inner.output_count()
    }

    fn input_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.shared.inner.input_fact(ix)
    }

    fn output_fact(&self, ix: usize) -> TractResult<&TypedFact> {
        self.shared.inner.output_fact(ix)
    }

    fn properties(&self) -> &HashMap<String, Arc<Tensor>> {
        self.shared.inner.properties()
    }

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
        self.shared.inner.typed_plan()
    }

    fn typed_model(&self) -> Option<&Arc<TypedModel>> {
        self.shared.inner.typed_model()
    }
}

/// A caller's view of a [`PooledRunnable`]: it holds no state of its own, and
/// each `run()` is one call to whichever worker is free.
#[derive(Clone, Debug)]
pub struct PooledStateHandle {
    runnable: PooledRunnable,
}

impl State for PooledStateHandle {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        self.runnable.call(inputs)
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.runnable
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Answers its input as is, and panics on a negative one. Spawns succeed
    /// until `spawnable` of them have.
    #[derive(Clone, Debug)]
    struct Panicky {
        spawned: Arc<AtomicUsize>,
        spawnable: usize,
        properties: HashMap<String, Arc<Tensor>>,
    }

    #[derive(Clone, Debug)]
    struct PanickyState(Panicky);

    impl Panicky {
        fn new(spawnable: usize) -> Panicky {
            let properties = [("tract.name".to_string(), rctensor0("panicky".to_string()))].into();
            Panicky { spawned: Arc::default(), spawnable, properties }
        }

        fn spawned(&self) -> usize {
            self.spawned.load(Ordering::SeqCst)
        }
    }

    impl Runnable for Panicky {
        fn spawn(&self) -> TractResult<Box<dyn State>> {
            ensure!(self.spawned.fetch_add(1, Ordering::SeqCst) < self.spawnable, "No more states");
            Ok(Box::new(PanickyState(self.clone())))
        }

        fn properties(&self) -> &HashMap<String, Arc<Tensor>> {
            &self.properties
        }

        fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
            None
        }

        fn typed_model(&self) -> Option<&Arc<TypedModel>> {
            None
        }
    }

    impl State for PanickyState {
        fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
            if inputs[0].cast_to_scalar::<f32>()? < 0. {
                panic!("negative input");
            }
            Ok(inputs)
        }

        fn runnable(&self) -> &dyn Runnable {
            &self.0
        }
    }

    fn call(pool: &PooledRunnable, x: f32) -> TractResult<f32> {
        let outputs = pool.run(tvec!(tensor0(x).into_tvalue()))?;
        outputs[0].cast_to_scalar::<f32>()
    }

    #[test]
    fn a_panicking_call_fails_and_its_worker_gets_a_fresh_state() -> TractResult<()> {
        use metrics_util::debugging::{DebugValue, DebuggingRecorder};
        let recorder = DebuggingRecorder::new();
        let inner = Panicky::new(usize::MAX);
        let pool = metrics::with_local_recorder(&recorder, || {
            PooledRunnable::wrap(Arc::new(inner.clone()), 2, 4)
        })?;
        assert_eq!(inner.spawned(), 2);
        for round in 0..3 {
            let e = call(&pool, -1.).unwrap_err();
            assert!(format!("{e}").contains("negative input"), "{e}");
            assert_eq!(call(&pool, round as f32)?, round as f32);
        }
        assert_eq!(inner.spawned(), 5);
        let panics = recorder.snapshotter().snapshot().into_vec().into_iter().find_map(
            |(key, _, _, value)| {
                (key.key().name() == "tract_pool_worker_panics_total").then_some(value)
            },
        );
        assert_eq!(panics, Some(DebugValue::Counter(3)));
        Ok(())
    }

    #[test]
    fn a_worker_failing_to_replace_its_state_stops() -> TractResult<()> {
        let inner = Panicky::new(1);
        let pool = PooledRunnable::wrap(Arc::new(inner.clone()), 1, 4)?;
        assert_eq!(call(&pool, 1.)?, 1.);
        assert!(call(&pool, -1.).is_err());
        // Queued before the worker stops, dropped with the queue; after, refused.
        for _ in 0..3 {
            assert!(call(&pool, 1.).is_err());
        }
        Ok(())
    }

    #[test]
    fn a_failing_spawn_fails_the_wrap() {
        assert!(PooledRunnable::wrap(Arc::new(Panicky::new(2)), 3, 4).is_err());
    }
}

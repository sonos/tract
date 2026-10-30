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
//! Dropping the runnable closes the queue and joins the workers: their
//! per-thread device state must be torn down before the libraries are.

use std::fmt::Debug;
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
        let labels: Vec<Label> =
            name.map(|name| Label::new("runnable", name.to_string())).into_iter().collect();
        Meters {
            queued: metrics::gauge!("tract_pool_queued", labels.clone()),
            rejected: metrics::counter!("tract_pool_rejected_total", labels.clone()),
            queue_wait: metrics::histogram!("tract_pool_queue_wait_seconds", labels.clone()),
            run: metrics::histogram!("tract_pool_run_seconds", labels),
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
                        work(state, &queued, &meters);
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
/// answer it, until the queue is closed.
fn work(mut state: Box<dyn State>, queued: &Mutex<Receiver<Call>>, meters: &Meters) {
    loop {
        let call = match queued.lock() {
            Ok(queued) => queued.recv(),
            Err(_) => return,
        };
        let Ok(call) = call else { return };
        meters.queued.decrement(1);
        meters.queue_wait.record(call.queued.elapsed());
        let start = Instant::now();
        let outputs = state.run(call.inputs);
        meters.run.record(start.elapsed());
        let _ = call.done.send(outputs);
    }
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

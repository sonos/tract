//! Dynamic micro-batching runtime that wraps the CUDA runtime.
//!
//! Registered as `cuda-batched`. It is transparent and hot-swappable by runtime
//! name: a model prepared through it behaves exactly like one prepared on `cuda`,
//! except that concurrent `run()` calls from independent caller threads are
//! gathered and executed as ONE batched device call.
//!
//! Motivation: the transducer decode loop fires one tiny predictor/joiner call
//! per token, per stream. With N concurrent streams that is N tiny serialized
//! CPU↔GPU round-trips per step (measured: ~360k ctx-switches/s, GPU busy with
//! launch overhead rather than work). Each model here already carries a symbolic
//! batch axis (the beam), so the fix is to grow that axis across streams: one
//! background worker per prepared model drains the queue of in-flight calls,
//! concatenates their inputs along the batch axis, runs the inner CUDA runnable
//! once, and scatters the row-slices back to each caller.
//!
//! Contract for a batchable model: every input and output must carry the batch
//! dimension on exactly one axis (a free symbol), and all other dims must be
//! identical across concurrent callers. Inputs/outputs with no symbolic axis are
//! treated as shared (broadcast once / replicated back). This holds for the
//! predictor and (after its `encoder_outputs` is reshaped to `[B,1024]`) the
//! joiner; it does NOT hold for the encoder (variable sequence length) — that
//! needs padding and is out of scope here.
//!
//! Knobs (env): `TRACT_BATCH_LINGER_US` (default 0) waits this long after the
//! first queued call to let more accumulate — the throughput↔latency dial;
//! `TRACT_BATCH_MAX` (default 256) caps batch size.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use tract_core::internal::*;

use crate::CudaRuntime;

/// Observability: number of batched device calls and total jobs served, since
/// process start. `mean batch = jobs / batches`.
pub static BATCHES: AtomicU64 = AtomicU64::new(0);
pub static JOBS: AtomicU64 = AtomicU64::new(0);

/// (batched device calls, total jobs served) — for `mean batch = jobs/batches`.
pub fn batch_stats() -> (u64, u64) {
    (BATCHES.load(Ordering::Relaxed), JOBS.load(Ordering::Relaxed))
}

/// Reply channel: the worker sends each caller its row-slices (owned tensors).
type Reply = Sender<TractResult<TVec<Tensor>>>;

struct Job {
    inputs: TVec<Tensor>,
    reply: Reply,
}

#[derive(Debug)]
struct BatchingCudaRuntime;

impl Runtime for BatchingCudaRuntime {
    fn name(&self) -> StaticName {
        "cuda-batched".into()
    }

    fn check(&self) -> TractResult<()> {
        CudaRuntime.check()
    }

    fn prepare_with_options(
        &self,
        model: TypedModel,
        options: &RunOptions,
    ) -> TractResult<Box<dyn Runnable>> {
        let inner = CudaRuntime.prepare_with_options(model, options)?;
        BatchingRunnable::wrap(inner)
    }
}

register_runtime!(BatchingCudaRuntime = BatchingCudaRuntime);

struct Shared {
    tx: Mutex<Sender<Job>>,
    model: Option<Arc<TypedModel>>,
    plan: Option<Arc<TypedSimplePlan>>,
}

#[derive(Clone)]
struct BatchingRunnable {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for BatchingRunnable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BatchingRunnable(cuda-batched)")
    }
}

/// The batch axis of a fact: its single symbolic axis, or `None` for a
/// shared/broadcast tensor (no symbolic axis).
fn detect_axis(fact: &TypedFact) -> TractResult<Option<usize>> {
    let symbolic: Vec<usize> = fact
        .shape
        .dims()
        .iter()
        .enumerate()
        .filter(|(_, d)| d.as_i64().is_none())
        .map(|(i, _)| i)
        .collect();
    match symbolic.len() {
        0 => Ok(None),
        1 => Ok(Some(symbolic[0])),
        _ => bail!("cuda-batched: model has {} symbolic axes in fact {:?}; need exactly one batch axis", symbolic.len(), fact),
    }
}

impl BatchingRunnable {
    fn wrap(inner: Box<dyn Runnable>) -> TractResult<Box<dyn Runnable>> {
        let model = inner.typed_model().cloned();
        let plan = inner.typed_plan().cloned();

        let mut batch_in = Vec::with_capacity(inner.input_count());
        for ix in 0..inner.input_count() {
            batch_in.push(detect_axis(inner.input_fact(ix)?)?);
        }
        let mut batch_out = Vec::with_capacity(inner.output_count());
        for ix in 0..inner.output_count() {
            batch_out.push(detect_axis(inner.output_fact(ix)?)?);
        }

        let linger_us: u64 = std::env::var("TRACT_BATCH_LINGER_US").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let max_batch: usize = std::env::var("TRACT_BATCH_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(256);

        let (tx, rx) = channel::<Job>();
        thread::Builder::new()
            .name("tract-batcher".into())
            .spawn(move || worker(inner, rx, batch_in, batch_out, linger_us, max_batch))?;

        Ok(Box::new(BatchingRunnable {
            shared: Arc::new(Shared { tx: Mutex::new(tx), model, plan }),
        }))
    }
}

impl Runnable for BatchingRunnable {
    fn spawn(&self) -> TractResult<Box<dyn State>> {
        let tx = self.shared.tx.lock().map_err(|_| format_err!("cuda-batched sender poisoned"))?.clone();
        Ok(Box::new(BatchingState { tx, runnable: self.clone() }))
    }

    fn typed_plan(&self) -> Option<&Arc<TypedSimplePlan>> {
        self.shared.plan.as_ref()
    }

    fn typed_model(&self) -> Option<&Arc<TypedModel>> {
        self.shared.model.as_ref()
    }
}

struct BatchingState {
    tx: Sender<Job>,
    runnable: BatchingRunnable,
}

impl std::fmt::Debug for BatchingState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BatchingState(cuda-batched)")
    }
}

impl State for BatchingState {
    fn run(&mut self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs: TVec<Tensor> = inputs.into_iter().map(|v| v.into_tensor()).collect();
        let (rtx, rrx) = channel();
        self.tx.send(Job { inputs, reply: rtx }).map_err(|_| format_err!("cuda-batched worker is gone"))?;
        let out = rrx.recv().map_err(|_| format_err!("cuda-batched worker dropped the reply"))??;
        Ok(out.into_iter().map(|t| t.into()).collect())
    }

    fn runnable(&self) -> &dyn Runnable {
        &self.runnable
    }

    fn freeze(&self) -> Box<dyn FrozenState> {
        Box::new(FrozenBatchingState { tx: self.tx.clone(), runnable: self.runnable.clone() })
    }
}

#[derive(Clone)]
struct FrozenBatchingState {
    tx: Sender<Job>,
    runnable: BatchingRunnable,
}

impl std::fmt::Debug for FrozenBatchingState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "FrozenBatchingState(cuda-batched)")
    }
}

impl FrozenState for FrozenBatchingState {
    fn unfreeze(&self) -> Box<dyn State> {
        Box::new(BatchingState { tx: self.tx.clone(), runnable: self.runnable.clone() })
    }
    fn input_count(&self) -> usize {
        self.runnable.input_count()
    }
    fn output_count(&self) -> usize {
        self.runnable.output_count()
    }
}

fn worker(
    inner: Box<dyn Runnable>,
    rx: Receiver<Job>,
    batch_in: Vec<Option<usize>>,
    batch_out: Vec<Option<usize>>,
    linger_us: u64,
    max_batch: usize,
) {
    loop {
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => return, // all senders dropped: the runnable was dropped
        };
        let mut jobs = vec![first];
        if linger_us > 0 {
            thread::sleep(Duration::from_micros(linger_us));
        }
        while jobs.len() < max_batch {
            match rx.try_recv() {
                Ok(j) => jobs.push(j),
                Err(_) => break,
            }
        }
        BATCHES.fetch_add(1, Ordering::Relaxed);
        JOBS.fetch_add(jobs.len() as u64, Ordering::Relaxed);
        match run_batch(&*inner, &jobs, &batch_in, &batch_out) {
            Ok(per_job) => {
                for (job, out) in jobs.into_iter().zip(per_job.into_iter()) {
                    let _ = job.reply.send(Ok(out));
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                for job in jobs {
                    let _ = job.reply.send(Err(format_err!("cuda-batched run failed: {msg}")));
                }
            }
        }
    }
}

fn run_batch(
    inner: &dyn Runnable,
    jobs: &[Job],
    batch_in: &[Option<usize>],
    batch_out: &[Option<usize>],
) -> TractResult<Vec<TVec<Tensor>>> {
    let n_in = jobs[0].inputs.len();

    // Per-job row count, read off the first input that carries a batch axis.
    let probe = batch_in.iter().position(|a| a.is_some());
    let Some(probe) = probe else {
        // No batchable axis anywhere: run each job independently (no win, but correct).
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            let inputs: TVec<TValue> = job.inputs.iter().cloned().map(|t| t.into()).collect();
            out.push(inner.run(inputs)?.into_iter().map(|v| v.into_tensor()).collect());
        }
        return Ok(out);
    };
    let probe_axis = batch_in[probe].unwrap();
    let rows: Vec<usize> = jobs.iter().map(|j| j.inputs[probe].shape()[probe_axis]).collect();

    // Concatenate inputs along their batch axis (shared inputs taken from job 0).
    let mut batched: TVec<TValue> = tvec!();
    for i in 0..n_in {
        match batch_in[i] {
            Some(ax) => {
                let refs: Vec<&Tensor> = jobs.iter().map(|j| &j.inputs[i]).collect();
                batched.push(Tensor::stack_tensors(ax, &refs)?.into());
            }
            None => batched.push(jobs[0].inputs[i].clone().into()),
        }
    }

    let out: TVec<TValue> = inner.run(batched)?;

    // Scatter each output's row-slices back to the originating jobs.
    let mut result: Vec<TVec<Tensor>> = (0..jobs.len()).map(|_| tvec!()).collect();
    for (o, out_value) in out.into_iter().enumerate() {
        match batch_out.get(o).copied().flatten() {
            Some(ax) => {
                let t = out_value.into_tensor();
                let mut off = 0;
                for (ji, &r) in rows.iter().enumerate() {
                    result[ji].push(t.slice(ax, off, off + r)?);
                    off += r;
                }
                ensure!(off == t.shape()[ax], "cuda-batched: output rows {} != batch {}", off, t.shape()[ax]);
            }
            None => {
                let t = out_value.into_tensor();
                for slot in result.iter_mut() {
                    slot.push(t.clone());
                }
            }
        }
    }
    Ok(result)
}

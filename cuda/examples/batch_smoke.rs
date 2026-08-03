//! Validates the `cuda-batched` runtime on a real transducer predictor.
//!
//! Usage: cargo run --release -p tract-cuda --example batch_smoke -- <predictor.nnef.tgz>
//!
//! Checks (a) correctness — a batched call returns the same logits as plain
//! `cuda`; (b) that concurrent calls actually get batched (mean batch > 1); and
//! (c) throughput of `cuda-batched` vs `cuda` as concurrency rises.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use tract_core::internal::*;
use tract_core::runtime::runtime_for_name;

const B: usize = 4; // beam width per call

/// Build zero inputs matching the model's input facts, substituting the
/// symbolic (batch) axis with B. Works for predictor and joiner alike.
fn inputs_for(r: &dyn Runnable) -> TVec<TValue> {
    let mut out = tvec!();
    for ix in 0..r.input_count() {
        let f = r.input_fact(ix).unwrap();
        let shape: Vec<usize> = f.shape.dims().iter().map(|d| d.as_i64().map(|v| v as usize).unwrap_or(B)).collect();
        out.push(Tensor::zero_dt(f.datum_type, &shape).unwrap().into());
    }
    out
}

fn throughput(runnable: &Arc<Box<dyn Runnable>>, n: usize, dur: Duration) -> f64 {
    let total = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|_| {
            let r = Arc::clone(runnable);
            let total = Arc::clone(&total);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut st = r.spawn().unwrap();
                while !stop.load(Ordering::Relaxed) {
                    st.run(inputs_for(&**r)).unwrap();
                    total.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    thread::sleep(dur);
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
    total.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()
}

fn main() -> TractResult<()> {
    let path = std::env::args().nth(1).expect("usage: batch_smoke <predictor.nnef.tgz>");
    let model = tract_nnef::nnef().model_for_path(&path)?.into_decluttered()?;

    let cuda = runtime_for_name("cuda")?.expect("cuda runtime not compiled in");
    let cuda_run: Arc<Box<dyn Runnable>> = Arc::new(cuda.prepare(model.clone())?);

    let batched = runtime_for_name("cuda-batched")?.expect("cuda-batched runtime not compiled in");
    let batched_run: Arc<Box<dyn Runnable>> = Arc::new(batched.prepare(model.clone())?);

    // (a) correctness
    let reference = cuda_run.run(inputs_for(&**cuda_run))?;
    let mut st = batched_run.spawn()?;
    let got = st.run(inputs_for(&**batched_run))?;
    for (i, (x, y)) in reference.iter().zip(got.iter()).enumerate() {
        x.close_enough(y, true).map_err(|e| format_err!("output {i} mismatch: {e}"))?;
    }
    eprintln!(
        "correctness OK: cuda-batched == cuda (output shapes {:?})",
        reference.iter().map(|t| t.shape().to_vec()).collect::<Vec<_>>()
    );

    // (b)+(c) throughput vs concurrency
    let dur = Duration::from_secs(3);
    println!("# threads | cuda calls/s | cuda-batched calls/s | speedup | mean_batch");
    for &n in &[1usize, 2, 4, 8, 16, 32, 64] {
        let c = throughput(&cuda_run, n, dur);
        let (b0, j0) = tract_cuda::batch_stats();
        let cb = throughput(&batched_run, n, dur);
        let (b1, j1) = tract_cuda::batch_stats();
        let mean_batch = (j1 - j0) as f64 / (b1 - b0).max(1) as f64;
        println!("{n:>9} | {c:>12.0} | {cb:>20.0} | {:>6.2}x | {mean_batch:>10.2}", cb / c.max(1.0));
    }
    Ok(())
}

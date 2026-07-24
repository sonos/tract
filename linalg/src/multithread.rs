use std::cell::RefCell;
#[cfg(feature = "multithread-mm")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};

#[cfg(feature = "multithread-mm")]
use rayon::{ThreadPool, ThreadPoolBuilder};

use tract_data::internal::TractResult;

#[derive(Debug, Clone, Default)]
pub enum Executor {
    #[default]
    SingleThread,
    #[cfg(feature = "multithread-mm")]
    MultiThread(Arc<ThreadPool>),
    /// Use rayon's GLOBAL thread pool — the one set up by
    /// `wasm_bindgen_rayon::init_thread_pool` on `wasm32-unknown-unknown`,
    /// or rayon's auto-initialised default on native.
    ///
    /// Exists because `Arc<rayon::ThreadPool>` cannot be constructed on
    /// `wasm32-unknown-unknown`: rayon's default `spawn_handler` calls
    /// `std::thread::spawn`, which is unsupported there. The only working
    /// route is rayon's global pool, accessed via `into_par_iter` directly.
    #[cfg(feature = "multithread-mm")]
    RayonGlobal,
}

impl Executor {
    #[cfg(feature = "multithread-mm")]
    pub fn multithread(n: usize) -> Executor {
        Executor::multithread_with_name(n, "tract-default")
    }

    #[cfg(feature = "multithread-mm")]
    pub fn multithread_with_name(n: usize, name: &str) -> Executor {
        let name = name.to_string();
        let pool = ThreadPoolBuilder::new()
            .thread_name(move |n| format!("{name}-{n}"))
            .num_threads(n)
            .build()
            .unwrap();
        Executor::MultiThread(Arc::new(pool))
    }
}

static DEFAULT_EXECUTOR: Mutex<Executor> = Mutex::new(Executor::SingleThread);

thread_local! {
    static TLS_EXECUTOR_OVERRIDE: RefCell<Option<Executor>> = Default::default();
}

pub fn current_tract_executor() -> Executor {
    if let Some(over_ride) = TLS_EXECUTOR_OVERRIDE.with_borrow(|tls| tls.clone()) {
        over_ride
    } else {
        DEFAULT_EXECUTOR.lock().unwrap().clone()
    }
}

pub fn set_default_executor(executor: Executor) {
    *DEFAULT_EXECUTOR.lock().unwrap() = executor;
}

pub fn multithread_tract_scope<R, F: FnOnce() -> R>(pool: Executor, f: F) -> R {
    let previous = TLS_EXECUTOR_OVERRIDE.replace(Some(pool));
    let result = f();
    TLS_EXECUTOR_OVERRIDE.set(previous);
    result
}

/// Threshold (in panels) below which the rayon MMM dispatcher skips
/// parallelism and runs inline single-threaded. Below this size,
/// per-call dispatch overhead (~5 µs native, ~50 µs wasm-bindgen-rayon
/// worker) exceeds the parallel speedup.
///
/// Default `64`. Tune higher for many-small-MMM workloads (mobile vision,
/// streaming RNN) or lower for transformer-class workloads where every MMM
/// is large. `0` disables the gate entirely (always thread).
#[cfg(feature = "multithread-mm")]
static THREADING_PANEL_THRESHOLD: AtomicUsize = AtomicUsize::new(64);

/// Read the current MMM panel-count threshold for the rayon path.
#[cfg(feature = "multithread-mm")]
pub fn current_threading_panel_threshold() -> usize {
    THREADING_PANEL_THRESHOLD.load(Ordering::Relaxed)
}

/// Set the MMM panel-count threshold for the rayon path. Default is `64`.
/// Pass `0` to thread regardless of size.
#[cfg(feature = "multithread-mm")]
pub fn set_threading_panel_threshold(panels: usize) {
    THREADING_PANEL_THRESHOLD.store(panels, Ordering::Relaxed);
}

/// Threshold (in tensor elements) below which [`par_chunks_mut`] skips
/// parallelism and runs its body inline single-threaded. Below this much work,
/// per-dispatch overhead exceeds the parallel speedup. Distinct from
/// `THREADING_PANEL_THRESHOLD`: this counts elements of work, not MMM panels.
///
/// Default `32768`. `0` disables the gate entirely (always thread).
#[cfg(feature = "multithread-mm")]
static THREADING_ELEMENT_THRESHOLD: AtomicUsize = AtomicUsize::new(32768);

/// Read the current element-count threshold for the row-parallel path.
#[cfg(feature = "multithread-mm")]
pub fn current_threading_element_threshold() -> usize {
    THREADING_ELEMENT_THRESHOLD.load(Ordering::Relaxed)
}

/// Set the element-count threshold for the row-parallel path. Default is
/// `32768`. Pass `0` to thread regardless of size.
#[cfg(feature = "multithread-mm")]
pub fn set_threading_element_threshold(elements: usize) {
    THREADING_ELEMENT_THRESHOLD.store(elements, Ordering::Relaxed);
}

/// Process `out` in parallel over its outer (row) axis, dispatching across the
/// executor installed by [`multithread_tract_scope`]. Falls back to a single
/// inline `f(0, out)` when the executor is single-threaded (including a
/// one-thread pool), when there are fewer than two rows, or when `total_elems`
/// is below [`current_threading_element_threshold`].
///
/// `out` is viewed as `out.len() / row_len` contiguous rows of `row_len`
/// elements (`row_len` must divide `out.len()`). Work is split only on row
/// boundaries, never inside a row, so any per-row reduction the closure runs
/// keeps its accumulation order and the output is bit-identical to the inline
/// path regardless of thread count.
///
/// The closure receives `(first_row, chunk)`: `chunk` is a contiguous block of
/// whole rows and `first_row` is the index of its first row within `out`, used
/// to index sibling buffers captured from the caller (e.g. an out-of-place
/// reduce whose input row is `reduced_dim` wide while `out` rows are width 1).
/// For such callers `total_elems` is the size of the data actually read, which
/// can exceed `out.len()`.
///
/// The signature is identical with or without the `multithread-mm` feature so
/// callers compile unchanged; without the feature the body is just `f(0, out)`.
pub fn par_chunks_mut<T: Send>(
    out: &mut [T],
    row_len: usize,
    total_elems: usize,
    f: impl Fn(usize, &mut [T]) -> TractResult<()> + Sync + Send,
) -> TractResult<()> {
    #[cfg(feature = "multithread-mm")]
    {
        use rayon::prelude::*;
        debug_assert!(row_len >= 1 && out.len() % row_len == 0);
        let n_rows = out.len() / row_len;
        if n_rows < 2 || total_elems < current_threading_element_threshold() {
            return f(0, out);
        }
        let run = |out: &mut [T]| -> TractResult<()> {
            let n_chunks = (4 * rayon::current_num_threads()).min(n_rows);
            let chunk_rows = n_rows.div_ceil(n_chunks);
            out.par_chunks_mut(chunk_rows * row_len)
                .enumerate()
                .try_for_each(|(i, chunk)| f(i * chunk_rows, chunk))
        };
        match current_tract_executor() {
            Executor::MultiThread(pool) if pool.current_num_threads() > 1 => {
                pool.install(|| run(out))
            }
            Executor::RayonGlobal => run(out),
            // SingleThread, or a one-thread MultiThread pool, runs inline serially.
            _ => f(0, out),
        }
    }
    #[cfg(not(feature = "multithread-mm"))]
    {
        let _ = (row_len, total_elems);
        f(0, out)
    }
}

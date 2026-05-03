use std::cell::RefCell;
#[cfg(feature = "multithread-mm")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};

#[cfg(feature = "multithread-mm")]
use rayon::{ThreadPool, ThreadPoolBuilder};

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

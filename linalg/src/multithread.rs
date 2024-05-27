use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use rayon::{ThreadPool, ThreadPoolBuilder};

static TRACT_THREAD_POOL: Mutex<Option<Arc<ThreadPool>>> = Mutex::new(None);

thread_local! {
    static TRACT_TLS_THREAD_POOL_OVERRIDE: RefCell<Option<Option<Arc<ThreadPool>>>> = Default::default();
}

pub fn tract_thread_pool() -> Option<Arc<ThreadPool>> {
    if let Some(over_ride) = TRACT_TLS_THREAD_POOL_OVERRIDE.with_borrow(|tls| tls.clone()) {
        over_ride
    } else {
        TRACT_THREAD_POOL.lock().unwrap().clone()
    }
}

pub fn set_tract_global_threads(n: usize) {
    let pool = ThreadPoolBuilder::new()
        .thread_name(|n| format!("tract-compute-{n}"))
        .num_threads(n)
        .build()
        .unwrap();
    *TRACT_THREAD_POOL.lock().unwrap() = Some(Arc::new(pool));
}

pub fn set_tract_global_threads_default() {
    set_tract_global_threads(num_cpus::get_physical())
}

pub fn tract_threads() -> usize {
    TRACT_THREAD_POOL.lock().unwrap().as_ref().map(|pool| pool.current_num_threads()).unwrap_or(0)
}

pub fn multithread_tract_scope<R, F: FnOnce() -> R>(n: usize, f: F) -> R {
    let pool = ThreadPoolBuilder::new()
        .thread_name(|n| format!("tract-compute-local-{n}"))
        .num_threads(n)
        .build()
        .unwrap();
    let previous = TRACT_TLS_THREAD_POOL_OVERRIDE.take();
    TRACT_TLS_THREAD_POOL_OVERRIDE.set(Some(Some(Arc::new(pool))));
    let result = f();
    TRACT_TLS_THREAD_POOL_OVERRIDE.set(previous);
    result
}

pub fn multithread_tract_scope_default<R, F: FnOnce() -> R>(f: F) -> R {
    multithread_tract_scope(num_cpus::get_physical(), f)
}

pub fn monothread_tract_scope<R, F: FnOnce() -> R>(f: F) -> R {
    let previous = TRACT_TLS_THREAD_POOL_OVERRIDE.take();
    TRACT_TLS_THREAD_POOL_OVERRIDE.set(Some(None));
    let result = f();
    TRACT_TLS_THREAD_POOL_OVERRIDE.set(previous);
    result
}

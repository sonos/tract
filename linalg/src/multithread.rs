use std::cell::RefCell;
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

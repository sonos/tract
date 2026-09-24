use std::cell::RefCell;
#[cfg(feature = "multithread-mm")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[allow(unused_imports)]
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "multithread-mm")]
use rayon::{ThreadPool, ThreadPoolBuilder};

#[cfg(feature = "multithread-mm")]
use tract_data::internal::vector_size;
use tract_data::internal::{Tensor, TensorView, TractResult, ensure};

use crate::BinFn;

/// Workload signature for adaptive chunking.
/// Captures the essential characteristics that affect optimal chunk size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WorkloadSig {
    row_len: usize,
    n_rows: usize,
    num_threads: usize,
}

/// Adaptive chunking state.
/// Stores learned optimal chunk_rows per workload signature.
#[cfg(feature = "multithread-mm")]
#[derive(Default)]
struct AdaptiveChunker {
    /// Maps workload signature -> optimal chunk_rows
    cache: Mutex<std::collections::HashMap<WorkloadSig, usize>>,
    /// Whether adaptive tuning is enabled
    enabled: AtomicBool,
}

#[cfg(feature = "multithread-mm")]
impl AdaptiveChunker {
    fn new() -> Self {
        Self { cache: Mutex::new(std::collections::HashMap::new()), enabled: AtomicBool::new(true) }
    }

    fn get_chunk_rows(&self, sig: WorkloadSig, threshold: usize, _row_len: usize) -> usize {
        if !self.enabled.load(Ordering::Relaxed) {
            return (threshold / sig.row_len).max(1);
        }
        let cache = self.cache.lock().unwrap();
        cache.get(&sig).copied().unwrap_or_else(|| (threshold / sig.row_len).max(1))
    }

    fn record(&self, sig: WorkloadSig, chunk_rows: usize) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut cache = self.cache.lock().unwrap();
        cache.insert(sig, chunk_rows);
    }

    fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }
}

/// Global adaptive chunker instance.
#[cfg(feature = "multithread-mm")]
static ADAPTIVE_CHUNKER: std::sync::OnceLock<AdaptiveChunker> = std::sync::OnceLock::new();

#[cfg(feature = "multithread-mm")]
fn adaptive_chunker() -> &'static AdaptiveChunker {
    ADAPTIVE_CHUNKER.get_or_init(AdaptiveChunker::new)
}

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
            let threshold = current_threading_element_threshold();
            let num_threads = rayon::current_num_threads();
            let sig = WorkloadSig { row_len, n_rows, num_threads };
            // Use adaptive chunker to get optimal chunk_rows
            let chunk_rows = adaptive_chunker().get_chunk_rows(sig, threshold, row_len);
            let n_chunks =
                n_rows.div_ceil(chunk_rows).min(n_rows).min(4 * rayon::current_num_threads());
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

/// How `b` maps onto the blocks [`par_bin`] splits `a` into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BShare {
    /// `b` is one `period`-element row that every block of `a` consumes in
    /// lockstep, as the unicast kernels do. Requires `b.len() == period`.
    Lockstep,
    /// `b` holds one scalar per block of `a`, as the by-scalar kernels do:
    /// block `i` reads `b[i]`. Requires `b.len() == a.len() / period`.
    PerBlock,
}

/// Apply the linalg binary kernel `eval_fn` to `a` in place, splitting the work
/// across the executor installed by [`multithread_tract_scope`].
///
/// `a` is `a.len() / period` blocks of `period` elements, `period` being how many
/// `a` elements one kernel call covers; `share` says how `b` lines up with those
/// blocks. A block is the coarsest unit a single call may span, because the
/// unicast kernels walk `b` in lockstep and index past its end once `a` runs
/// beyond one period. Inside a block the split lands on `vector_size()`-element
/// boundaries, which keeps `a`'s and `b`'s offsets congruent modulo the kernel
/// alignment — `unicast_with_alignment` hard-asserts that congruence.
/// `OptBinUnicast::check_b_alignement` is what guarantees `period` itself is such
/// a multiple whenever there is more than one block.
///
/// An empty `a` is a no-op: the kernels cannot take one, as `as_slice_mut` would
/// build a slice from the null data pointer.
///
/// Both tensors must have natural C-order strides, which the callers' stride
/// guards establish, and plain storage, which holds because no linalg binary
/// kernel is registered for a datum type that lacks it. The byte offsets computed
/// here are only correct under both.
///
/// These kernels are pure elementwise, so no chunk boundary can change a result:
/// the output is bit-identical to the serial path at any thread count.
pub fn par_bin(
    eval_fn: &BinFn,
    a: &mut Tensor,
    b: &Tensor,
    period: usize,
    share: BShare,
) -> TractResult<()> {
    if a.len() == 0 {
        return Ok(());
    }
    ensure!(
        a.len().is_multiple_of(period),
        "par_bin: period {period} does not divide a.len() {}",
        a.len()
    );
    let n_blocks = a.len() / period;
    match share {
        BShare::Lockstep => ensure!(
            b.len() == period,
            "par_bin: Lockstep wants b.len() {} == period {period}",
            b.len()
        ),
        BShare::PerBlock => ensure!(
            b.len() == n_blocks,
            "par_bin: PerBlock wants b.len() {} == {n_blocks} blocks",
            b.len()
        ),
    }

    let a_item = a.datum_type().size_of() as isize;
    let b_item = b.datum_type().size_of() as isize;
    let a = &*a;
    // `len` elements of block `block`, starting `offset` elements into it. Both
    // tensors are naturally strided, so a's flat element index is
    // `block * period + offset` and b's is `offset` (Lockstep) or `block`
    // (PerBlock).
    let call = |block: usize, offset: usize, len: usize| -> TractResult<()> {
        static STRIDES: [isize; 1] = [1];
        let a_shape = [len];
        let (b_offset, b_shape) = match share {
            BShare::Lockstep => (offset as isize * b_item, [len]),
            BShare::PerBlock => (block as isize * b_item, [1]),
        };
        let a_offset = (block * period + offset) as isize * a_item;
        // `offset + len <= period` and blocks are disjoint, so the byte range
        // `[a_offset, a_offset + len * a_item)` is disjoint across every
        // (block, offset) the dispatch below enumerates. That is what makes the
        // concurrent writes through these views non-aliasing; keep it true if the
        // chunk arithmetic changes.
        unsafe {
            let mut a_chunk = TensorView::from_bytes(a, a_offset, &a_shape, &STRIDES);
            let b_chunk = TensorView::from_bytes(b, b_offset, &b_shape, &STRIDES);
            eval_fn(&mut a_chunk, &b_chunk)
        }
    };

    #[cfg(feature = "multithread-mm")]
    {
        use rayon::prelude::*;
        // Threshold first: reading the executor takes a global lock, and a graph
        // has hundreds of these nodes sitting below the threshold.
        if a.len() >= current_threading_element_threshold() {
            let executor = current_tract_executor();
            let nth = match &executor {
                Executor::MultiThread(pool) => pool.current_num_threads(),
                Executor::RayonGlobal => rayon::current_num_threads(),
                Executor::SingleThread => 1,
            };
            if nth > 1 {
                let per_block = (4 * nth).div_ceil(n_blocks).max(1);
                let chunk = period.div_ceil(per_block).next_multiple_of(vector_size()).min(period);
                let per_block = period.div_ceil(chunk);
                if n_blocks * per_block > 1 {
                    let run = || {
                        (0..n_blocks * per_block).into_par_iter().try_for_each(|i| {
                            let offset = (i % per_block) * chunk;
                            call(i / per_block, offset, chunk.min(period - offset))
                        })
                    };
                    return match executor {
                        Executor::MultiThread(pool) => pool.install(run),
                        _ => run(),
                    };
                }
            }
        }
    }

    // A single block is the whole tensor, so hand the kernel the natural view
    // rather than a rank-1 one: fewer shapes for a future kernel to have to
    // tolerate on the overwhelmingly common path.
    if n_blocks == 1 {
        return eval_fn(&mut a.view(), &b.view());
    }
    (0..n_blocks).try_for_each(|block| call(block, 0, period))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn run_par_chunks<T: Send + Clone>(
        out: &mut [T],
        row_len: usize,
        total_elems: usize,
        f: impl Fn(usize, &mut [T]) -> TractResult<()> + Sync + Send,
    ) -> TractResult<()> {
        par_chunks_mut(out, row_len, total_elems, f)
    }

    #[test]
    fn par_chunks_mut_inline_when_single_row() {
        let mut out = vec![0u32; 10];
        let called = std::sync::Arc::new(std::sync::Mutex::new(0));
        let c = called.clone();
        run_par_chunks(&mut out, 10, 10, move |first, chunk| {
            assert_eq!(first, 0);
            assert_eq!(chunk.len(), 10);
            *c.lock().unwrap() += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(*called.lock().unwrap(), 1);
    }

    #[test]
    fn par_chunks_mut_inline_when_below_threshold() {
        let mut out = vec![0u32; 100];
        let called = std::sync::Arc::new(std::sync::Mutex::new(0));
        let c = called.clone();
        // 5 rows * 10 = 50 elements, below default 32768 threshold
        // But inline path passes the FULL slice (100 elements)
        run_par_chunks(&mut out, 10, 50, move |first, chunk| {
            assert_eq!(first, 0);
            assert_eq!(chunk.len(), 100); // inline passes full slice
            *c.lock().unwrap() += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(*called.lock().unwrap(), 1);
    }

    #[test]
    fn par_chunks_mut_chunks_correctly() {
        let mut out = vec![0u32; 1000];
        let called = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let c = called.clone();
        // 100 rows of 10 elements = 1000 total, well above threshold
        run_par_chunks(&mut out, 10, 1000, move |first, chunk| {
            c.lock().unwrap().push((first, chunk.len()));
            Ok(())
        })
        .unwrap();
        let calls = called.lock().unwrap();
        assert!(!calls.is_empty());
        // Sum of chunk lengths should equal total
        let sum: usize = calls.iter().map(|(_, len)| *len).sum();
        assert_eq!(sum, 1000);
        // Each chunk should be multiple of row_len (10)
        for (_, len) in calls.iter() {
            assert_eq!(*len % 10, 0);
        }
    }

    #[test]
    fn par_chunks_mut_chunk_size_respects_threshold() {
        // row_len = 100, threshold = 32768
        // chunk_rows = 32768 / 100 = 327, so ~327 rows per chunk
        let row_len = 100;
        let total_elems = 100000; // 1000 rows
        let mut out = vec![0u32; total_elems];
        let called = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let c = called.clone();
        run_par_chunks(&mut out, row_len, total_elems, move |first, chunk| {
            c.lock().unwrap().push((first, chunk.len()));
            Ok(())
        })
        .unwrap();
        let calls = called.lock().unwrap();
        for (_, len) in calls.iter() {
            // Each chunk should have at least threshold elements (except possibly last)
            // chunk_rows = 32768/100 = 327, so chunk_size = 327*100 = 32700
            assert!(*len >= 32700 || *len == total_elems % (327 * 100));
        }
    }
}

/// Tune the adaptive chunker for a specific workload by trying different chunk sizes
/// and measuring execution time. This should be called during a warmup phase.
#[cfg(feature = "multithread-mm")]
pub fn tune_par_chunks_mut<T: Send + Clone>(
    out: &mut [T],
    row_len: usize,
    total_elems: usize,
    f: impl Fn(usize, &mut [T]) -> TractResult<()> + Sync + Send,
) -> TractResult<usize> {
    use rayon::prelude::*;

    debug_assert!(row_len >= 1 && out.len() % row_len == 0);
    let n_rows = out.len() / row_len;
    if n_rows < 2 || total_elems < current_threading_element_threshold() {
        return Ok(1);
    }

    let threshold = current_threading_element_threshold();
    let num_threads = rayon::current_num_threads();
    let sig = WorkloadSig { row_len, n_rows, num_threads };

    // Try different chunk sizes and measure
    let candidates = vec![
        (threshold / row_len).max(1), // Static heuristic
        (threshold / row_len / 2).max(1),
        (threshold / row_len * 2).max(1),
        (threshold / row_len * 4).max(1),
        n_rows.div_ceil(4 * num_threads).max(1), // rayon default
        n_rows,                                  // 1 chunk per row
    ];

    let mut best_chunk_rows = (threshold / row_len).max(1);
    let mut best_time = Duration::MAX;

    for chunk_rows in candidates {
        let n_chunks = n_rows.div_ceil(chunk_rows).min(n_rows).min(4 * num_threads);
        let actual_chunk_rows = n_rows.div_ceil(n_chunks);

        // Warmup run
        let mut temp_out = out.to_vec();
        let _ =
            temp_out.par_chunks_mut(actual_chunk_rows * row_len).try_for_each(|chunk| f(0, chunk));

        // Timed run
        let start = Instant::now();
        let _ = out.par_chunks_mut(actual_chunk_rows * row_len).try_for_each(|chunk| f(0, chunk));
        let elapsed = start.elapsed();

        if elapsed < best_time {
            best_time = elapsed;
            best_chunk_rows = actual_chunk_rows;
        }
    }

    // Record the best chunk_rows
    adaptive_chunker().record(sig, best_chunk_rows);

    Ok(best_chunk_rows)
}

/// Enable or disable adaptive chunking globally.
#[cfg(feature = "multithread-mm")]
pub fn set_adaptive_chunking(enabled: bool) {
    if let Some(chunker) = ADAPTIVE_CHUNKER.get() {
        if enabled {
            chunker.enable();
        } else {
            chunker.disable();
        }
    }
}

/// Check if adaptive chunking is enabled.
#[cfg(feature = "multithread-mm")]
pub fn is_adaptive_chunking_enabled() -> bool {
    ADAPTIVE_CHUNKER.get().map(|c| c.enabled).unwrap_or(false)
}

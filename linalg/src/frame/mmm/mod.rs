#[macro_use]
mod macros;

pub mod cost_model;
#[macro_use]
pub(crate) mod fuse;
pub(crate) mod input_store;
pub(crate) mod kernel;
#[macro_use]
pub(crate) mod panel_extract;
mod scratch;
mod storage;

#[cfg(test)]
#[macro_use]
pub mod tests;

use crate::multithread::Executor;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Debug;
use tract_data::internal::*;

pub use cost_model::*;
pub use fuse::*;
pub use input_store::*;
pub use kernel::*;
pub use panel_extract::*;
pub use scratch::*;
pub use storage::*;

pub fn no_prefetch(_ptr: *const u8, _len: usize) {}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum ImplementationQuality {
    /// Individual operations are emulated by individual conversion (f16->f32->f16)
    Dreadful,
    /// Rust scalar operation (with whatever optimisation the compiler manages)
    Generic,
    /// Implicit vectorization (e.g. Rust code, some unrolled loops, explicit template instantiations for small constant)
    RustOptimized,
    /// Explicit vectorization (e.g. intrinsics vector code)
    TargetOptimized,
    /// Hand optimized (assembly)
    ManuallyOptimized,
}

impl ImplementationQuality {
    pub fn best_to_worst() -> &'static [ImplementationQuality] {
        use ImplementationQuality::*;
        &[ManuallyOptimized, TargetOptimized, RustOptimized, Generic, Dreadful]
    }

    pub fn cost(&self) -> usize {
        ImplementationQuality::best_to_worst().iter().position(|x| x == self).unwrap()
    }
}

impl PartialOrd for ImplementationQuality {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(usize::from(*self).cmp(&usize::from(*other)))
    }
}

impl From<ImplementationQuality> for usize {
    fn from(value: ImplementationQuality) -> Self {
        value.cost()
    }
}

pub trait MatMatMul: Debug + dyn_clone::DynClone + Send + Sync + std::any::Any {
    fn name(&self) -> &str;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    fn quality(&self) -> ImplementationQuality;
    fn dynamic_boost(&self) -> isize;

    /// Whether this kernel is runnable on the current CPU (platform feature
    /// gate, e.g. FEAT_DotProd for the SDOT i8 kernel).
    fn is_supported_here(&self) -> bool;

    #[allow(clippy::type_complexity)]
    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)];

    fn internal_type(&self) -> DatumType;

    unsafe fn c_view(&self, m_axis: Option<usize>, n_axis: Option<usize>) -> OutputStoreSpec;
    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec;

    fn can_fuse(&self, spec: &FusedSpec) -> bool;

    fn stores(&self) -> Cow<'_, [DatumType]>;

    unsafe fn run(&self, m: usize, n: usize, non_linear: &[FusedSpec]) -> TractResult<()> {
        unsafe {
            let mut scratch = self.allocate_scratch_space();
            self.run_with_scratch_space(m, n, &mut *scratch, non_linear)
        }
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace>;
    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool;
    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()>;
}

dyn_clone::clone_trait_object!(MatMatMul);

impl PartialEq for Box<dyn MatMatMul> {
    fn eq(&self, other: &Box<dyn MatMatMul>) -> bool {
        self.name() == other.name()
    }
}
impl Eq for Box<dyn MatMatMul> {}

impl std::hash::Hash for Box<dyn MatMatMul> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl<K: MatMatMulKer> MatMatMul for K {
    fn name(&self) -> &str {
        self.name()
    }
    fn mr(&self) -> usize {
        self.mr()
    }
    fn nr(&self) -> usize {
        self.nr()
    }

    fn quality(&self) -> ImplementationQuality {
        MatMatMulKer::quality(self)
    }

    fn dynamic_boost(&self) -> isize {
        MatMatMulKer::dynamic_boost(self)
    }

    fn is_supported_here(&self) -> bool {
        MatMatMulKer::is_supported_here(self)
    }

    fn packings(&self) -> &[(Box<dyn MMMInputFormat>, Box<dyn MMMInputFormat>)] {
        self.packings()
    }

    fn internal_type(&self) -> DatumType {
        K::Acc::datum_type()
    }

    fn can_fuse(&self, spec: &FusedSpec) -> bool {
        self.can_fuse(spec)
    }

    unsafe fn c_view(&self, m_axis: Option<usize>, n_axis: Option<usize>) -> OutputStoreSpec {
        OutputStoreSpec::View { m_axis, n_axis, mr: self.mr(), nr: self.nr() }
    }

    unsafe fn c_from_data_and_strides(
        &self,
        item_size: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> OutputStoreSpec {
        OutputStoreSpec::Strides {
            row_byte_stride: row_stride * item_size as isize,
            col_byte_stride: col_stride * item_size as isize,
            mr: self.mr(),
            nr: self.nr(),
        }
    }

    fn stores(&self) -> Cow<'_, [DatumType]> {
        self.stores()
    }

    unsafe fn allocate_scratch_space(&self) -> Box<dyn ScratchSpace> {
        Box::<ScratchSpaceImpl<K::Acc>>::default()
    }

    unsafe fn can_use_scratch_space(&self, scratch: &dyn ScratchSpace) -> bool {
        scratch.downcast_ref::<ScratchSpaceImpl<K::Acc>>().is_some()
    }

    unsafe fn run_with_scratch_space(
        &self,
        m: usize,
        n: usize,
        scratch: &mut dyn ScratchSpace,
        non_linear: &[FusedSpec],
    ) -> TractResult<()> {
        // Every AddMatMul must pass panels packed the way the named packing index
        // expects; a mismatch reads the panels at the wrong stride and runs off the
        // buffer. Guard it here so any caller — not just OptMatMul — is caught.
        #[cfg(debug_assertions)]
        {
            use crate::pack::PackedFormat;
            // Only raw PackedFormat panels can be read at the wrong stride; exotic
            // inputs (lazy im2col, block-quant) materialise panels in the kernel's
            // format via panel_bytes, so a differing wrapper type is fine. When both
            // sides are PackedFormat, require same element type and row count
            // (tolerating alignment/padding, but not an f16-vs-f32 element-size swap).
            fn compatible(expected: &dyn MMMInputFormat, got: &dyn MMMInputFormat) -> bool {
                if expected.dyn_eq(got) {
                    return true;
                }
                match (expected.downcast_ref::<PackedFormat>(), got.downcast_ref::<PackedFormat>())
                {
                    (Some(e), Some(g)) => e.dt == g.dt && e.r == g.r,
                    _ => true,
                }
            }
            for spec in non_linear {
                if let FusedSpec::AddMatMul { a, b, packing } = spec {
                    let (pa, pb) = &self.packings()[*packing];
                    debug_assert!(
                        compatible(&**pa, a.format()),
                        "A packed as {:?} but {} packing {packing} expects {pa:?}",
                        a.format(),
                        self.name(),
                    );
                    debug_assert!(
                        compatible(&**pb, b.format()),
                        "B packed as {:?} but {} packing {packing} expects {pb:?}",
                        b.format(),
                        self.name(),
                    );
                }
            }
        }
        unsafe {
            let scratch = scratch
                .downcast_mut::<ScratchSpaceImpl<K::Acc>>()
                .context("Wrong scratch space type")?;
            scratch.prepare(self, m, n, non_linear)?;
            if n == 1 && self.nr() == 1 {
                run_with_scratch_space_vec(self, m, scratch, non_linear)
            } else {
                let (mut prefer_col, mut prefer_row) = (0, 0);
                for uop in non_linear.iter() {
                    if let Some(col) = uop.prefer_col_outer() {
                        prefer_col = col as usize;
                        prefer_row = (!col) as usize;
                    }
                }
                // k drives the single-thread cache-block size; read it from the
                // first AddMatMul's packed input (0 if none → max block).
                let k = non_linear
                    .iter()
                    .find_map(|f| match f {
                        FusedSpec::AddMatMul { a, .. } => Some(a.k()),
                        _ => None,
                    })
                    .unwrap_or(0);
                if prefer_col > prefer_row {
                    run_with_scratch_space_col_outer(self, m, n, k, scratch, non_linear)
                } else {
                    run_with_scratch_space_row_outer(self, m, n, k, scratch, non_linear)
                }
            }
        }
    }
}

unsafe fn run_with_scratch_space_vec<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => scratch.run_in_tls_scope(|scratch, tls| {
                for ia in 0..m.divceil(ker.mr()) {
                    scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                }
                TractResult::Ok(())
            }),
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => chunked_dispatch_rayon(
                Some(&pool),
                m.divceil(ker.mr()),
                1,
                |ia_start, ia_end, _, _| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => {
                chunked_dispatch_rayon(None, m.divceil(ker.mr()), 1, |ia_start, ia_end, _, _| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                        }
                        TractResult::Ok(())
                    })
                })
            }
        }
    }
}

/// Upper bound on the inner (L2-resident) panel-block edge (matches the
/// multithread `chunk_grid` default).
const ST_BLK_MAX: usize = 16;

/// Upper bound on the outer (L3-resident) super-block edge. 4× the inner cap so
/// an L3 several times larger than L2 can hold a meaningfully bigger super-block.
const ST_BLK_L3_MAX: usize = 64;

/// Panel-block working-set budget (bytes) from a detected cache size: a fraction
/// `num/den` of the cache (leaving room for the C accumulator tile + packing
/// metadata), clamped to a sane range. `0` (cache unknown) ⇒ `fallback`, which
/// is kept small so the block ≈ the naive loop and can never over-block a cache
/// it can't see. Sizes come from the shared [`crate::cache`] probe.
fn tier_budget_bytes(cache_bytes: usize, num: usize, den: usize, fallback: usize) -> usize {
    if cache_bytes == 0 {
        fallback
    } else {
        (cache_bytes * num / den).clamp(64 * 1024, 64 * 1024 * 1024)
    }
}

/// Inner tier: ~a third of L2 (private per perf-core), 256 KiB fallback.
fn l2_block_budget_bytes() -> usize {
    tier_budget_bytes(crate::cache::cache_info().l2, 1, 3, 256 * 1024)
}

/// Outer tier: `(llc_bytes, budget_bytes)` — the raw last-level-cache size and the
/// fraction of it a single thread may budget for the outer super-block — but only
/// when an L3/LLC larger than L2 is detected (otherwise an outer tier just
/// duplicates the inner one). `None` ⇒ no outer tier; the walk stays single-level
/// (identical to before). The raw size is returned alongside the budget so the
/// caller can check whether the working set even spills the cache before blocking.
fn l3_block_budget_bytes() -> Option<(usize, usize)> {
    use crate::cache::LlcKind;
    let (bytes, kind) = crate::cache::last_level_cache()?;
    // Dedicated cluster L3: ~half. A shared System-Level Cache is contended by the
    // GPU/NPU/display, so we can't assume residency of lines they keep evicting —
    // budget it to ~a quarter.
    let (num, den) = match kind {
        LlcKind::Dedicated => (1, 2),
        LlcKind::SystemLevel => (1, 4),
    };
    Some((bytes, tier_budget_bytes(bytes, num, den, 0)))
}

/// Cache-adaptive panel-block edge for a given byte budget: large enough to
/// amortise streaming, small enough that the block's A+B sub-panels
/// (`~blk·(mr+nr)·k·elem_bytes`) stay cache-resident at the given `k`. Capped at
/// `cap`; the floor of 1 degrades exactly to the naive loop, so an unknown/small
/// cache can never over-block (regression-safe).
#[inline]
fn block_edge_for(
    budget: usize,
    mr: usize,
    nr: usize,
    k: usize,
    elem_bytes: usize,
    cap: usize,
) -> usize {
    if k == 0 {
        return cap;
    }
    let per_blk = ((mr + nr) * k * elem_bytes.max(1)).max(1);
    (budget / per_blk).clamp(1, cap)
}

/// Whether inner (L2) blocking captures reuse the naive stream cannot, given the
/// operand the walk re-streams — A (the m side, `panels·r = m_panels·mr`) for a
/// column-outer order, B (the n side) for a row-outer one. If that streamed
/// operand already fits L2 it is re-read from cache, not DRAM, so reordering
/// tiles buys no reuse and only disturbs the prefetchers; only when it spills L2
/// does the block save re-fetches. Mirrors [`outer_tier_pays`] for the inner
/// tier, keyed on the streamed operand rather than the whole working set.
fn inner_tier_pays(panels: usize, r: usize, k: usize, elem_bytes: usize, l2_bytes: usize) -> bool {
    let streamed = panels.saturating_mul(r).saturating_mul(k).saturating_mul(elem_bytes);
    l2_bytes > 0 && streamed > l2_bytes
}

/// Inner (L2) panel-block edge, or `usize::MAX` (single block, i.e. the naive
/// stream) when the streamed operand already fits L2 (see [`inner_tier_pays`]).
/// The budget is **cache-size derived** (not a hard-coded constant), so it is
/// correct across hardware.
#[inline]
fn st_block_edge(
    mr: usize,
    nr: usize,
    k: usize,
    elem_bytes: usize,
    m_panels: usize,
    n_panels: usize,
    col_outer: bool,
) -> usize {
    let (panels, r) = if col_outer { (m_panels, mr) } else { (n_panels, nr) };
    if !inner_tier_pays(panels, r, k, elem_bytes, crate::cache::cache_info().l2) {
        return usize::MAX;
    }
    block_edge_for(l2_block_budget_bytes(), mr, nr, k, elem_bytes, ST_BLK_MAX)
}

/// Whether an L3 outer super-block can capture reuse the inner (L2) tier cannot.
/// It only can when the packed working set (`A + B ≈ (m·mr + n·nr)·k·elem`)
/// actually spills the last-level cache: if both operands already fit, they stay
/// resident across the sweep regardless of traversal order, so the reorder buys
/// no reuse and only disturbs the hardware prefetchers — a measured net loss on
/// small models that never leave L3 (voicecom_float on jetson-orin-nx, +15.6%).
/// This is exactly the precondition the outer tier was introduced for ("a grid
/// that exceeds L2 still re-fetches A/B from DRAM"); without the check the tier
/// also engages on grids that never leave the LLC.
fn outer_tier_pays(
    m_panels: usize,
    n_panels: usize,
    mr: usize,
    nr: usize,
    k: usize,
    elem_bytes: usize,
    llc_bytes: usize,
) -> bool {
    let working_set = m_panels
        .saturating_mul(mr)
        .saturating_add(n_panels.saturating_mul(nr))
        .saturating_mul(k)
        .saturating_mul(elem_bytes);
    llc_bytes > 0 && working_set > llc_bytes
}

/// Outer (L3) super-block edge, or `usize::MAX` (one block over the whole grid,
/// i.e. no outer tier) when no usable L3 is detected or the working set already
/// fits it (see [`outer_tier_pays`]). Never smaller than the inner edge `inner`.
#[inline]
fn st_outer_block_edge(
    mr: usize,
    nr: usize,
    k: usize,
    elem_bytes: usize,
    inner: usize,
    m_panels: usize,
    n_panels: usize,
) -> usize {
    let Some((llc, budget)) = l3_block_budget_bytes() else { return usize::MAX };
    if !outer_tier_pays(m_panels, n_panels, mr, nr, k, elem_bytes, llc) {
        return usize::MAX;
    }
    block_edge_for(budget, mr, nr, k, elem_bytes, ST_BLK_L3_MAX).max(inner)
}

/// Visit every `(ia, ib)` tile of the `m_panels × n_panels` grid exactly once,
/// blocked two levels deep: an outer `blk_outer` super-block (L3-resident) holds
/// inner `blk` blocks (L2-resident). `col_outer` selects the within-block inner
/// order (B-reuse vs A-reuse). When `blk_outer >= max(m,n)` the outer loop runs
/// once and this is exactly the single-level inner walk. Pure tile reordering ⇒
/// no result changes; extracted so the nesting can be unit-tested independently
/// of the kernel.
#[inline]
fn for_each_blocked_tile(
    m_panels: usize,
    n_panels: usize,
    blk: usize,
    blk_outer: usize,
    col_outer: bool,
    mut f: impl FnMut(usize, usize) -> TractResult<()>,
) -> TractResult<()> {
    let blk = blk.max(1);
    let blk_outer = blk_outer.max(blk);
    let mut jb3 = 0;
    while jb3 < n_panels {
        let jb3_end = jb3.saturating_add(blk_outer).min(n_panels);
        let mut ja3 = 0;
        while ja3 < m_panels {
            let ja3_end = ja3.saturating_add(blk_outer).min(m_panels);
            let mut jb = jb3;
            while jb < jb3_end {
                let jb_end = (jb + blk).min(jb3_end);
                let mut ja = ja3;
                while ja < ja3_end {
                    let ja_end = (ja + blk).min(ja3_end);
                    if col_outer {
                        for ib in jb..jb_end {
                            for ia in ja..ja_end {
                                f(ia, ib)?;
                            }
                        }
                    } else {
                        for ia in ja..ja_end {
                            for ib in jb..jb_end {
                                f(ia, ib)?;
                            }
                        }
                    }
                    ja = ja_end;
                }
                jb = jb_end;
            }
            ja3 = ja3_end;
        }
        jb3 = jb3_end;
    }
    Ok(())
}

/// Single-thread tile walk over the `m_panels × n_panels` grid, blocked into
/// cache-sized panel blocks for locality (the naive nested loop re-streams the
/// whole inner operand per outer panel at large k; the multithread path already
/// blocks this way via `chunk_grid`). Two tiers: an inner L2-resident block and,
/// where an L3 is detected, an outer L3-resident super-block. Reordering
/// independent tiles changes no result — bit-exact with the naive loop.
#[inline]
unsafe fn run_single_thread_blocked<K: MatMatMulKer>(
    ker: &K,
    m_panels: usize,
    n_panels: usize,
    k: usize,
    col_outer: bool,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        let elem = K::Acc::datum_type().size_of();
        let blk = st_block_edge(ker.mr(), ker.nr(), k, elem, m_panels, n_panels, col_outer);
        let blk_outer = st_outer_block_edge(ker.mr(), ker.nr(), k, elem, blk, m_panels, n_panels);
        scratch.run_in_tls_scope(|scratch, tls| {
            for_each_blocked_tile(m_panels, n_panels, blk, blk_outer, col_outer, |ia, ib| {
                scratch.run_one_tile(ker, non_linear, tls, ia, ib)
            })
        })
    }
}

unsafe fn run_with_scratch_space_col_outer<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    k: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => run_single_thread_blocked(
                ker,
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                k,
                true,
                scratch,
                non_linear,
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => chunked_dispatch_rayon(
                Some(&pool),
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ib in ib_start..ib_end {
                            for ia in ia_start..ia_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => chunked_dispatch_rayon(
                None,
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ib in ib_start..ib_end {
                            for ia in ia_start..ia_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
        }
    }
}

unsafe fn run_with_scratch_space_row_outer<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    k: usize,
    scratch: &mut ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => run_single_thread_blocked(
                ker,
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                k,
                false,
                scratch,
                non_linear,
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => chunked_dispatch_rayon(
                Some(&pool),
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            for ib in ib_start..ib_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => chunked_dispatch_rayon(
                None,
                m.divceil(ker.mr()),
                n.divceil(ker.nr()),
                |ia_start, ia_end, ib_start, ib_end| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            for ib in ib_start..ib_end {
                                scratch.run_one_tile(ker, non_linear, tls, ia, ib)?;
                            }
                        }
                        TractResult::Ok(())
                    })
                },
            ),
        }
    }
}

/// Chunk grid for the 2D dispatch.
///
/// Mirrors ggml's `mul_mat` heuristic (`ggml/src/ggml-cpu/ggml-cpu.c:1378-1398`):
///  * 16-tile panel chunks by default;
///  * 64-tile chunks when one dimension is 1 (vec / vec-mat);
///  * fallback to "block-per-thread along the longer axis" when the natural
///    grid would have fewer than `4·nth` chunks.
///
/// Returns `(nchunks_m, nchunks_n, dr_m, dr_n)`.
#[cfg(feature = "multithread-mm")]
fn chunk_grid(n_panels_m: usize, n_panels_n: usize, nth: usize) -> (usize, usize, usize, usize) {
    let chunk_size = if n_panels_m == 1 || n_panels_n == 1 { 64 } else { 16 };
    let mut nchunks_m = n_panels_m.div_ceil(chunk_size);
    let mut nchunks_n = n_panels_n.div_ceil(chunk_size);
    if nchunks_m * nchunks_n < 4 * nth {
        if n_panels_m > n_panels_n {
            nchunks_m = nth;
            nchunks_n = 1;
        } else {
            nchunks_m = 1;
            nchunks_n = nth;
        }
    }
    let dr_m = n_panels_m.div_ceil(nchunks_m).max(1);
    let dr_n = n_panels_n.div_ceil(nchunks_n).max(1);
    (nchunks_m, nchunks_n, dr_m, dr_n)
}

/// 2D chunked dispatcher across the (m_panels × n_panels) grid for the
/// rayon path. Replaces a 1D `into_par_iter` over a single panel axis.
/// Better-utilises threads on small/skewed shapes where one dimension has
/// fewer panels than there are workers.
///
/// The closure receives **chunk bounds** (`ia_start, ia_end, ib_start, ib_end`),
/// not per-tile indices. This lets the caller amortise per-worker setup
/// (e.g. `ScratchSpaceImpl::run_in_tls_scope`) across all tiles in the
/// chunk, mirroring #2206 for the multi-threaded path. The closure is
/// invoked exactly once per rayon work item (and once total when the
/// small-graph fallback path is taken).
///
/// `pool`:
///   * `Some(p)` with `p.current_num_threads() > 1` → scoped via `p.install`
///     (native, custom pool path).
///   * `Some(p)` with single-thread pool, or `None` → dispatched via
///     `into_par_iter` directly, which uses rayon's GLOBAL pool. This is
///     the only working path on `wasm32-unknown-unknown` via
///     `wasm_bindgen_rayon::init_thread_pool`.
#[cfg(feature = "multithread-mm")]
unsafe fn chunked_dispatch_rayon<F>(
    pool: Option<&rayon::ThreadPool>,
    n_panels_m: usize,
    n_panels_n: usize,
    run_chunk: F,
) -> TractResult<()>
where
    F: Fn(usize, usize, usize, usize) -> TractResult<()> + Sync,
{
    use rayon::prelude::*;
    if n_panels_m == 0 || n_panels_n == 0 {
        return Ok(());
    }
    if n_panels_m * n_panels_n < crate::multithread::current_threading_panel_threshold() {
        // Below the threading threshold: run the whole grid as a single chunk
        // on the calling thread. Closure handles its own TLS scope.
        return run_chunk(0, n_panels_m, 0, n_panels_n);
    }
    let use_global = pool.is_none_or(|p| p.current_num_threads() <= 1);
    let body = || {
        let nth = rayon::current_num_threads();
        let (nchunks_m, nchunks_n, dr_m, dr_n) = chunk_grid(n_panels_m, n_panels_n, nth);
        let total = nchunks_m * nchunks_n;
        (0..total).into_par_iter().try_for_each(|idx| {
            let im = idx % nchunks_m;
            let in_ = idx / nchunks_m;
            let ia_start = im * dr_m;
            let ia_end = (ia_start + dr_m).min(n_panels_m);
            let ib_start = in_ * dr_n;
            let ib_end = (ib_start + dr_n).min(n_panels_n);
            run_chunk(ia_start, ia_end, ib_start, ib_end)
        })
    };
    if use_global { body() } else { pool.unwrap().install(body) }
}

#[cfg(test)]
mod blocked_walk_tests {
    use super::*;
    use std::collections::HashSet;

    fn collect(
        m: usize,
        n: usize,
        blk: usize,
        blk_outer: usize,
        col_outer: bool,
    ) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for_each_blocked_tile(m, n, blk, blk_outer, col_outer, |ia, ib| {
            v.push((ia, ib));
            Ok(())
        })
        .unwrap();
        v
    }

    /// Every grid tile is visited exactly once, for both inner orders and a
    /// range of (blk, blk_outer) — single-tier (outer = MAX), two-tier, and
    /// degenerate edges. Coverage being a permutation is what makes the walk
    /// bit-exact with the naive loop.
    #[test]
    fn covers_every_tile_once() {
        for &(m, n) in &[(1, 1), (3, 5), (16, 16), (40, 7), (7, 40), (80, 80)] {
            for &blk in &[1, 3, 16] {
                for &blk_outer in &[blk, blk + 1, 64, usize::MAX] {
                    for &col_outer in &[false, true] {
                        let tiles = collect(m, n, blk, blk_outer, col_outer);
                        assert_eq!(tiles.len(), m * n, "m={m} n={n} blk={blk} outer={blk_outer}");
                        let set: HashSet<_> = tiles.iter().copied().collect();
                        assert_eq!(
                            set.len(),
                            m * n,
                            "duplicate tiles m={m} n={n} blk={blk} outer={blk_outer}"
                        );
                        for ia in 0..m {
                            for ib in 0..n {
                                assert!(set.contains(&(ia, ib)), "missing ({ia},{ib})");
                            }
                        }
                    }
                }
            }
        }
    }

    /// With no outer tier (blk_outer = MAX) the two-tier walk must emit the exact
    /// same order as the original single-level blocked loop — guarantees the L3
    /// path is a pure no-op on hardware without a detectable L3.
    #[test]
    fn outer_max_matches_single_level() {
        for &(m, n) in &[(40, 7), (80, 80), (13, 29)] {
            for &blk in &[1, 4, 16] {
                for &col_outer in &[false, true] {
                    let two_tier = collect(m, n, blk, usize::MAX, col_outer);
                    let mut single = Vec::new();
                    let mut jb = 0;
                    while jb < n {
                        let jb_end = (jb + blk).min(n);
                        let mut ja = 0;
                        while ja < m {
                            let ja_end = (ja + blk).min(m);
                            if col_outer {
                                for ib in jb..jb_end {
                                    for ia in ja..ja_end {
                                        single.push((ia, ib));
                                    }
                                }
                            } else {
                                for ia in ja..ja_end {
                                    for ib in jb..jb_end {
                                        single.push((ia, ib));
                                    }
                                }
                            }
                            ja = ja_end;
                        }
                        jb = jb_end;
                    }
                    assert_eq!(two_tier, single, "m={m} n={n} blk={blk} col_outer={col_outer}");
                }
            }
        }
    }

    /// The outer tier engages only when the packed working set spills the LLC.
    /// A grid that already fits stays single-level (the reorder buys no reuse and
    /// only hurts prefetch — the voicecom_float/Orin regression).
    #[test]
    fn outer_tier_gated_on_working_set_spilling_llc() {
        let llc = 2 * 1024 * 1024; // 2 MiB, f32 (elem = 4)
        // Small grid: (64·8 + 8·8)·64·4 ≈ 144 KiB ⇒ fits ⇒ no outer tier.
        assert!(!outer_tier_pays(64, 8, 8, 8, 64, 4, llc));
        // Large grid: (256·8 + 256·8)·256·4 ≈ 4 MiB ⇒ spills ⇒ engage.
        assert!(outer_tier_pays(256, 256, 8, 8, 256, 4, llc));
        // A boundary working set equal to the LLC does not spill it.
        assert!(!outer_tier_pays(1, 0, llc, 0, 1, 1, llc));
        // Unknown LLC (0) never engages, whatever the grid.
        assert!(!outer_tier_pays(4096, 4096, 8, 8, 4096, 4, 0));
        // k = 0 (empty reduction) has no working set ⇒ never engages.
        assert!(!outer_tier_pays(4096, 4096, 8, 8, 0, 4, llc));
    }

    /// Inner blocking engages only when the operand the walk re-streams — A for a
    /// column-outer order, B for a row-outer one — spills L2. A streamed operand
    /// that fits is re-read from cache, so blocking only hurts prefetch.
    #[test]
    fn inner_tier_gated_on_streamed_operand_spilling_l2() {
        let l2 = 1024 * 1024; // 1 MiB, f32 (elem = 4)
        // inception Conv2d_4a_3x3 grid (16×12 kernel), k=720.
        // col_outer streams A (m side, panels=12 r=16): 12·16·720·4 ≈ 540 KiB ⇒ fits.
        assert!(!inner_tier_pays(12, 16, 720, 4, l2));
        // row_outer streams B (n side, panels=421 r=12): 421·12·720·4 ≈ 14.5 MiB ⇒ spills.
        assert!(inner_tier_pays(421, 12, 720, 4, l2));
        // A large square (m side, panels=256 r=16, k=512): 256·16·512·4 ≈ 8 MiB ⇒ spills.
        assert!(inner_tier_pays(256, 16, 512, 4, l2));
        // Undetectable L2 (0) never engages — degrades to the naive loop.
        assert!(!inner_tier_pays(4096, 16, 4096, 4, 0));
        // k = 0 (empty reduction) has no working set.
        assert!(!inner_tier_pays(4096, 16, 0, 4, l2));
    }
}

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
mod select;
mod storage;

#[cfg(test)]
#[macro_use]
pub mod tests;

use crate::multithread::Executor;
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::Range;
use tract_data::internal::*;

pub use cost_model::*;
pub use fuse::*;
pub use input_store::*;
pub use kernel::*;
pub use panel_extract::*;
pub use scratch::*;
pub use select::*;
pub use storage::*;

pub fn no_prefetch(_ptr: *const u8, _len: usize) {}

pub trait MatMatMul: Debug + dyn_clone::DynClone + Send + Sync + std::any::Any {
    fn name(&self) -> &str;
    fn mr(&self) -> usize;
    fn nr(&self) -> usize;

    /// Architecture this kernel is written for, `None` for the generic Rust every target
    /// builds. What [`retain_best`] compares before anything else: a kernel written for the
    /// machine at hand supersedes a portable one whatever their instruction sets.
    fn arch(&self) -> Option<crate::isa::Arch>;

    /// Whether the kernel computes its accumulator type by converting every operation to
    /// another type, for a machine whose hardware has none. Orders of magnitude off a real
    /// kernel, and always the only thing on offer where it is declared, so selection ignores
    /// it and benches skip it.
    fn emulated(&self) -> bool;

    /// The preference this kernel's author spelled out, before the instruction-set default.
    fn boost(&self) -> isize;

    /// Where this kernel sits against the siblings of its own kind, which [`retain_best`]
    /// weighs once [`Self::arch`] has not separated them: the level of the instruction set it
    /// is written for, plus whatever a measurement said that level gets wrong. A kernel
    /// written for a more capable set outranks one written for a less capable one by default;
    /// a declared boost is how an exception is spelled, and must be big enough to cross the
    /// levels it disagrees with. Never encode a preference in [`Self::runnable`] — that
    /// silently skips the kernel's tests as well.
    fn preference(&self) -> isize;

    /// Whether a machine with this instruction set could execute the kernel: the architecture is
    /// the one it is written for, and the set offers every feature it declares. Takes the machine
    /// rather than reading the host, so the same question serves dispatch and an audit of what
    /// another architecture would run.
    ///
    /// It says nothing about whether this build assembled the body — see [`Self::built`]. A
    /// kernel can be runnable on a machine and still be a stub here.
    fn runnable_on(&self, isa: &crate::isa::IsaSet) -> bool;

    /// Whether this kernel can be executed here at all: this build compiled it
    /// ([`Self::built`]) and the running CPU has the instruction set it declares
    /// ([`Self::runnable_on`] against the probed set).
    ///
    /// Runnability only, never preference: this answers "would executing the kernel fault",
    /// and the mmm test bodies gate on it, so a kernel that lies here has no test coverage at
    /// all on the hosts it lies on. Say a kernel is worse than its sibling with
    /// [`Self::preference`] instead.
    fn runnable(&self) -> bool;

    /// Whether this build compiled the kernel's body at all. False for a foreign arch's
    /// kernel, which is metadata around a stub that bails when called.
    fn built(&self) -> bool;

    /// What the instruction set must offer for this kernel to run here.
    fn isa(&self) -> crate::isa::IsaReq;

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

    fn arch(&self) -> Option<crate::isa::Arch> {
        MatMatMulKer::arch(self)
    }

    fn emulated(&self) -> bool {
        MatMatMulKer::emulated(self)
    }

    fn boost(&self) -> isize {
        MatMatMulKer::boost(self)
    }

    fn preference(&self) -> isize {
        MatMatMulKer::preference(self)
    }

    fn runnable_on(&self, isa: &crate::isa::IsaSet) -> bool {
        MatMatMulKer::runnable_on(self, isa)
    }

    fn runnable(&self) -> bool {
        MatMatMulKer::runnable(self)
    }

    fn built(&self) -> bool {
        MatMatMulKer::built(self)
    }

    fn isa(&self) -> crate::isa::IsaReq {
        MatMatMulKer::isa(self)
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
                // k drives the cache-block size; read it from the first
                // AddMatMul's packed input (0 if none → max block).
                let k = non_linear
                    .iter()
                    .find_map(|f| match f {
                        FusedSpec::AddMatMul { a, .. } => Some(a.k()),
                        _ => None,
                    })
                    .unwrap_or(0);
                run_with_scratch_space_2d(
                    self,
                    m,
                    n,
                    k,
                    prefer_col > prefer_row,
                    scratch,
                    non_linear,
                )
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
                ker.mr(),
                ker.nr(),
                |ia_start, ia_end, _, _, _| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                        }
                        TractResult::Ok(())
                    })
                },
            ),
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => chunked_dispatch_rayon(
                None,
                m.divceil(ker.mr()),
                1,
                ker.mr(),
                ker.nr(),
                |ia_start, ia_end, _, _, _| {
                    scratch.run_in_tls_scope(|scratch, tls| {
                        for ia in ia_start..ia_end {
                            scratch.run_one_tile(ker, non_linear, tls, ia, 0)?;
                        }
                        TractResult::Ok(())
                    })
                },
            ),
        }
    }
}

/// Upper bound on the inner (L2-resident) panel-block edge.
const BLK_MAX: usize = 16;

/// Upper bound on the outer (L3-resident) super-block edge. 4× the inner cap so
/// an L3 several times larger than L2 can hold a meaningfully bigger super-block.
const BLK_L3_MAX: usize = 64;

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
/// fraction of it the outer super-block may budget — but only when an L3/LLC larger
/// than L2 is detected (otherwise an outer tier just duplicates the inner one).
/// `None` ⇒ no outer tier; the walk stays single-level. The raw size is returned
/// alongside the budget so the caller can check whether the working set even
/// spills the cache before blocking. Both numbers are for the *whole* cache;
/// concurrent walkers each get a share (see [`outer_block_edge`]).
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
///
/// `l2_share` is how many rectangles concurrently share this L2, so each walker
/// may only assume its slice of it — like [`outer_block_edge`]'s `llc_share`, but
/// bounded by the L2's physical sharing degree rather than the thread count. On a
/// core-private L2 (`l2_share == 1`) this is the whole cache, unchanged; on a
/// cluster-shared L2 (Cortex-A9/A53) it prevents sibling rectangles from evicting
/// one another's blocks.
#[inline]
#[allow(clippy::too_many_arguments)]
fn inner_block_edge(
    mr: usize,
    nr: usize,
    k: usize,
    elem_bytes: usize,
    m_panels: usize,
    n_panels: usize,
    col_outer: bool,
    l2_share: usize,
) -> usize {
    let (panels, r) = if col_outer { (m_panels, mr) } else { (n_panels, nr) };
    let share = l2_share.max(1);
    if !inner_tier_pays(panels, r, k, elem_bytes, crate::cache::cache_info().l2 / share) {
        return usize::MAX;
    }
    block_edge_for(l2_block_budget_bytes() / share, mr, nr, k, elem_bytes, BLK_MAX)
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

/// Outer (L3) super-block edge, or `usize::MAX` (one block over the whole
/// rectangle, i.e. no outer tier) when no usable L3 is detected or the working set
/// already fits it (see [`outer_tier_pays`]). Never smaller than the inner edge
/// `inner`.
///
/// `llc_share` is how many rectangles are walked concurrently: the LLC is shared,
/// so each walker may only assume its slice of it. Sizing every chunk of a
/// multi-threaded dispatch against the whole LLC would have them evict each
/// other's super-blocks.
#[inline]
#[allow(clippy::too_many_arguments)]
fn outer_block_edge(
    mr: usize,
    nr: usize,
    k: usize,
    elem_bytes: usize,
    inner: usize,
    m_panels: usize,
    n_panels: usize,
    llc_share: usize,
) -> usize {
    let Some((llc, budget)) = l3_block_budget_bytes() else { return usize::MAX };
    let share = llc_share.max(1);
    if !outer_tier_pays(m_panels, n_panels, mr, nr, k, elem_bytes, llc / share) {
        return usize::MAX;
    }
    block_edge_for(budget / share, mr, nr, k, elem_bytes, BLK_L3_MAX).max(inner)
}

/// Visit every `(ia, ib)` tile of the `m × n` panel rectangle exactly once,
/// blocked two levels deep: an outer `blk_outer` super-block (L3-resident) holds
/// inner `blk` blocks (L2-resident). `col_outer` selects the within-block inner
/// order (B-reuse vs A-reuse). When `blk_outer` spans the whole rectangle the
/// outer loop runs once and this is exactly the single-level inner walk. Pure
/// tile reordering ⇒ no result changes; extracted so the nesting can be
/// unit-tested independently of the kernel.
#[inline]
fn for_each_blocked_tile(
    m: Range<usize>,
    n: Range<usize>,
    blk: usize,
    blk_outer: usize,
    col_outer: bool,
    mut f: impl FnMut(usize, usize) -> TractResult<()>,
) -> TractResult<()> {
    let blk = blk.max(1);
    let blk_outer = blk_outer.max(blk);
    let mut jb3 = n.start;
    while jb3 < n.end {
        let jb3_end = jb3.saturating_add(blk_outer).min(n.end);
        let mut ja3 = m.start;
        while ja3 < m.end {
            let ja3_end = ja3.saturating_add(blk_outer).min(m.end);
            let mut jb = jb3;
            while jb < jb3_end {
                let jb_end = jb.saturating_add(blk).min(jb3_end);
                let mut ja = ja3;
                while ja < ja3_end {
                    let ja_end = ja.saturating_add(blk).min(ja3_end);
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

/// Tile walk over one panel rectangle — the whole grid on the single-thread
/// path, one dispatch chunk on the rayon path — blocked into cache-sized panel
/// blocks for locality (the naive nested loop re-streams the whole inner operand
/// per outer panel at large k). Two tiers: an inner L2-resident block and, where
/// an L3 is detected, an outer L3-resident super-block sized for one of
/// `llc_share` concurrent walkers. Both tiers are gated on the rectangle's own
/// extents, so a rectangle whose streamed operand already fits cache walks
/// exactly the naive order. Reordering independent tiles changes no result —
/// bit-exact with the naive loop at any chunking.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn run_blocked<K: MatMatMulKer>(
    ker: &K,
    m: Range<usize>,
    n: Range<usize>,
    k: usize,
    col_outer: bool,
    llc_share: usize,
    scratch: &ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        let elem = K::Acc::datum_type().size_of();
        let (mr, nr) = (ker.mr(), ker.nr());
        let (m_panels, n_panels) = (m.len(), n.len());
        let l2_share = llc_share.min(crate::cache::cache_info().l2_sharers_or_one());
        let blk = inner_block_edge(mr, nr, k, elem, m_panels, n_panels, col_outer, l2_share);
        let blk_outer = outer_block_edge(mr, nr, k, elem, blk, m_panels, n_panels, llc_share);
        scratch.run_in_tls_scope(|scratch, tls| {
            for_each_blocked_tile(m, n, blk, blk_outer, col_outer, |ia, ib| {
                scratch.run_one_tile(ker, non_linear, tls, ia, ib)
            })
        })
    }
}

/// Run the whole `m × n` output over the executor currently installed: as one
/// rectangle when single-threaded, else split into the chunk grid
/// [`chunk_grid`] picks. `col_outer` selects the tile order inside a rectangle
/// (B-reuse vs A-reuse), from the fused ops' preference. `k` is only used to size
/// the cache blocking.
unsafe fn run_with_scratch_space_2d<K: MatMatMulKer>(
    ker: &K,
    m: usize,
    n: usize,
    k: usize,
    col_outer: bool,
    scratch: &ScratchSpaceImpl<K::Acc>,
    non_linear: &[FusedSpec],
) -> TractResult<()> {
    unsafe {
        let (m_panels, n_panels) = (m.divceil(ker.mr()), n.divceil(ker.nr()));
        #[cfg(feature = "multithread-mm")]
        let chunk = |ia_start, ia_end, ib_start, ib_end, concurrency| {
            run_blocked(
                ker,
                ia_start..ia_end,
                ib_start..ib_end,
                k,
                col_outer,
                concurrency,
                scratch,
                non_linear,
            )
        };
        match crate::multithread::current_tract_executor() {
            Executor::SingleThread => {
                run_blocked(ker, 0..m_panels, 0..n_panels, k, col_outer, 1, scratch, non_linear)
            }
            #[cfg(feature = "multithread-mm")]
            Executor::MultiThread(pool) => {
                chunked_dispatch_rayon(Some(&pool), m_panels, n_panels, ker.mr(), ker.nr(), chunk)
            }
            #[cfg(feature = "multithread-mm")]
            Executor::RayonGlobal => {
                chunked_dispatch_rayon(None, m_panels, n_panels, ker.mr(), ker.nr(), chunk)
            }
        }
    }
}

/// Chunks per thread the 2D dispatch aims for on a part whose last-level cache
/// can absorb the re-reads (see [`chunks_per_thread`]). Above one, rayon can steal
/// work when threads progress unevenly — a contended core, an E-core on a
/// big.LITTLE part, a chunk carrying more border tiles — so a straggler costs at
/// most its share rather than the whole grid's tail. Each extra chunk also
/// re-reads a band of the packed operands, and that cost grows as `sqrt(chunks)`
/// against a linear gain in slack, which is what keeps this small.
#[cfg(feature = "multithread-mm")]
const CHUNKS_PER_THREAD: usize = 4;

/// Smallest last-level cache that makes the extra chunks worth their re-reads.
/// Below this a part has only a small cluster L2 and no LLC behind it
/// (Cortex-A7/A9/A53), so the band a chunk re-reads comes from DRAM every time.
#[cfg(feature = "multithread-mm")]
const CHUNK_SLACK_LLC_BYTES: usize = 2 * 1024 * 1024;

/// Chunks per thread [`chunk_grid`] aims for on this machine.
///
/// An extra chunk buys load-balance slack and costs one more pass over a packed
/// operand. That pass is a cache hit on a part with a last-level cache big enough
/// to hold the band and a DRAM round trip on one without, so the slack is only
/// worth taking above [`CHUNK_SLACK_LLC_BYTES`]; below it the dispatch takes one
/// chunk per thread.
///
/// `TRACT_MMM_CHUNKS_PER_THREAD` overrides the choice. Resolved once, like the
/// cache probe it reads, so a later change to the variable has no effect.
#[cfg(feature = "multithread-mm")]
fn chunks_per_thread() -> usize {
    use std::sync::OnceLock;
    static CPT: OnceLock<usize> = OnceLock::new();
    *CPT.get_or_init(|| {
        if let Some(n) =
            std::env::var("TRACT_MMM_CHUNKS_PER_THREAD").ok().and_then(|v| v.trim().parse().ok())
        {
            return usize::max(n, 1);
        }
        let llc = crate::cache::last_level_cache()
            .map(|(bytes, _)| bytes)
            .unwrap_or_else(|| crate::cache::cache_info().l2);
        if llc >= CHUNK_SLACK_LLC_BYTES { CHUNKS_PER_THREAD } else { 1 }
    })
}

/// Chunk grid for the 2D dispatch: `(nchunks_m, nchunks_n, dr_m, dr_n)`.
///
/// Aims for [`chunks_per_thread`]`· nth` chunks and shapes them to minimise how
/// often the packed operands are re-read: a chunk covering `dr_m × dr_n` panels
/// reads `dr_m·mr·k` of A and `dr_n·nr·k` of B, so over the whole grid A is read
/// `nchunks_n` times and B `nchunks_m` times. Minimising
/// `nchunks_n·m + nchunks_m·n` at a fixed chunk count puts
/// `nchunks_m = sqrt(chunks · m / n)` — chunks as square as the operands'
/// extents, rather than a band across one axis.
///
/// Cache locality *inside* a chunk is [`run_blocked`]'s job, not this function's;
/// chunk count therefore tracks the thread count and not a cache size.
///
/// Both panel counts must be non-zero; the dispatcher returns early on an empty
/// grid.
#[cfg(feature = "multithread-mm")]
fn chunk_grid(
    n_panels_m: usize,
    n_panels_n: usize,
    mr: usize,
    nr: usize,
    nth: usize,
) -> (usize, usize, usize, usize) {
    let chunks = (chunks_per_thread() * nth).max(1);
    let (m, n) = (n_panels_m * mr, (n_panels_n * nr).max(1));
    let nchunks_m = (chunks.saturating_mul(m) / n).isqrt().clamp(1, n_panels_m);
    let nchunks_n = (chunks / nchunks_m).clamp(1, n_panels_n);
    let nchunks_m = (chunks / nchunks_n).clamp(1, n_panels_m);
    let dr_m = n_panels_m.div_ceil(nchunks_m);
    let dr_n = n_panels_n.div_ceil(nchunks_n);
    // Recount from the edges: `div_ceil` can make the last chunk of an axis land
    // entirely outside the grid, and an empty work item is a wasted dispatch.
    (n_panels_m.div_ceil(dr_m), n_panels_n.div_ceil(dr_n), dr_m, dr_n)
}

/// Dispatch the `m_panels × n_panels` panel grid across the rayon path, split into
/// the 2D chunk grid [`chunk_grid`] picks. Grids below
/// [`crate::multithread::current_threading_panel_threshold`] run whole on the
/// calling thread instead.
///
/// The closure receives **chunk bounds** (`ia_start, ia_end, ib_start, ib_end`)
/// plus the number of chunks running concurrently, not per-tile indices. Chunk
/// bounds let it amortise per-worker setup (e.g.
/// `ScratchSpaceImpl::run_in_tls_scope`) over all the tiles in the chunk; the
/// concurrency lets it size shared-cache blocking against the share it actually
/// gets. The closure is invoked exactly once per rayon work item, and once in
/// total with a concurrency of 1 on the below-threshold path.
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
    mr: usize,
    nr: usize,
    run_chunk: F,
) -> TractResult<()>
where
    F: Fn(usize, usize, usize, usize, usize) -> TractResult<()> + Sync,
{
    use rayon::prelude::*;
    if n_panels_m == 0 || n_panels_n == 0 {
        return Ok(());
    }
    if n_panels_m * n_panels_n < crate::multithread::current_threading_panel_threshold() {
        // Below the threading threshold: run the whole grid as a single chunk
        // on the calling thread. Closure handles its own TLS scope.
        return run_chunk(0, n_panels_m, 0, n_panels_n, 1);
    }
    let use_global = pool.is_none_or(|p| p.current_num_threads() <= 1);
    let body = || {
        let nth = rayon::current_num_threads();
        let (nchunks_m, nchunks_n, dr_m, dr_n) = chunk_grid(n_panels_m, n_panels_n, mr, nr, nth);
        let total = nchunks_m * nchunks_n;
        let concurrency = nth.min(total);
        (0..total).into_par_iter().try_for_each(|idx| {
            let im = idx % nchunks_m;
            let in_ = idx / nchunks_m;
            let ia_start = im * dr_m;
            let ia_end = (ia_start + dr_m).min(n_panels_m);
            let ib_start = in_ * dr_n;
            let ib_end = (ib_start + dr_n).min(n_panels_n);
            run_chunk(ia_start, ia_end, ib_start, ib_end, concurrency)
        })
    };
    if use_global { body() } else { pool.unwrap().install(body) }
}

#[cfg(test)]
mod blocked_walk_tests {
    use super::*;
    use std::collections::HashSet;

    fn collect(
        m: Range<usize>,
        n: Range<usize>,
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

    /// Every tile of the rectangle is visited exactly once, for both inner orders
    /// and a range of (blk, blk_outer) — single-tier (outer = MAX), two-tier, and
    /// degenerate edges. Coverage being a permutation is what makes the walk
    /// bit-exact with the naive loop. Offset rectangles are the dispatch chunks.
    #[test]
    fn covers_every_tile_once() {
        for &(m, n) in &[(1, 1), (3, 5), (16, 16), (40, 7), (7, 40), (80, 80)] {
            for &(m0, n0) in &[(0, 0), (3, 11)] {
                // usize::MAX is how both tiers say "do not block"; on an offset
                // rectangle the edge arithmetic must not overflow past the end.
                for &blk in &[1, 3, 16, usize::MAX] {
                    for &blk_outer in &[blk, blk.saturating_add(1), 64, usize::MAX] {
                        for &col_outer in &[false, true] {
                            let tiles = collect(m0..m0 + m, n0..n0 + n, blk, blk_outer, col_outer);
                            assert_eq!(
                                tiles.len(),
                                m * n,
                                "m={m} n={n} blk={blk} outer={blk_outer}"
                            );
                            let set: HashSet<_> = tiles.iter().copied().collect();
                            assert_eq!(
                                set.len(),
                                m * n,
                                "duplicate tiles m={m} n={n} blk={blk} outer={blk_outer}"
                            );
                            for ia in m0..m0 + m {
                                for ib in n0..n0 + n {
                                    assert!(set.contains(&(ia, ib)), "missing ({ia},{ib})");
                                }
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
                    let two_tier = collect(0..m, 0..n, blk, usize::MAX, col_outer);
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

    /// Grids, kernel aspect ratios and thread counts worth checking the chunk
    /// grid against: skewed both ways, square, prime-ish, and the degenerate
    /// single-panel cases.
    #[cfg(feature = "multithread-mm")]
    const GRIDS: &[(usize, usize)] = &[
        (1, 1),
        (1, 5),
        (5, 1),
        (2, 3),
        (3, 3),
        (16, 96),
        (96, 16),
        (17, 17),
        (64, 64),
        (32, 384),
        (128, 128),
        (1, 4096),
        (4096, 1),
        (9, 1000),
    ];

    #[cfg(feature = "multithread-mm")]
    const RATIOS: &[(usize, usize)] = &[(8, 8), (16, 4), (32, 32), (64, 1)];

    /// The four numbers `chunk_grid` returns must tile the panel grid exactly:
    /// `chunked_dispatch_rayon` turns them into work items, so an empty chunk is
    /// a wasted dispatch, an overlap would double-compute a tile, and a gap would
    /// leave part of C uninitialised.
    #[cfg(feature = "multithread-mm")]
    #[test]
    fn chunk_grid_tiles_the_panel_grid() {
        for &(m, n) in GRIDS {
            for &(mr, nr) in RATIOS {
                for nth in [1usize, 2, 3, 4, 6, 8, 16, 64] {
                    let (cm, cn, dr_m, dr_n) = chunk_grid(m, n, mr, nr, nth);
                    let ctx = format!("{m}x{n} panels, {mr}x{nr} kernel, {nth} threads");
                    let mut seen = vec![false; m * n];
                    for idx in 0..cm * cn {
                        let (im, in_) = (idx % cm, idx / cm);
                        let (a0, a1) = (im * dr_m, (im * dr_m + dr_m).min(m));
                        let (b0, b1) = (in_ * dr_n, (in_ * dr_n + dr_n).min(n));
                        assert!(a0 < a1 && b0 < b1, "empty chunk {idx} in {ctx}");
                        for ia in a0..a1 {
                            for ib in b0..b1 {
                                assert!(!seen[ia * n + ib], "tile ({ia},{ib}) twice in {ctx}");
                                seen[ia * n + ib] = true;
                            }
                        }
                    }
                    assert!(seen.iter().all(|s| *s), "tile left out in {ctx}");
                }
            }
        }
    }

    /// Enough chunks to keep every thread fed, whenever the grid has that many
    /// panels to go round. Recounting the chunks from the edges is what makes this
    /// hold: naming more chunks than the edges cover would idle the difference.
    #[cfg(feature = "multithread-mm")]
    #[test]
    fn chunk_grid_feeds_every_thread() {
        for &(m, n) in GRIDS {
            for &(mr, nr) in RATIOS {
                for nth in [1usize, 2, 3, 4, 6, 8, 16, 64] {
                    let (cm, cn, ..) = chunk_grid(m, n, mr, nr, nth);
                    assert!(
                        cm * cn >= nth.min(m * n),
                        "{cm}x{cn} chunks for {nth} threads on {m}x{n} panels"
                    );
                }
            }
        }
    }

    /// The grid is shaped to minimise packed-operand re-reads: A is read once per
    /// column of chunks and B once per row, so `nchunks_n·m + nchunks_m·n` is what
    /// the shape trades off. It must never cost more than a band across either
    /// axis at the same chunk count, which on a square grid costs 1.5x.
    #[cfg(feature = "multithread-mm")]
    #[test]
    fn chunk_grid_shape_beats_a_band_on_operand_traffic() {
        let traffic = |cm: usize, cn: usize, m: usize, n: usize| cn * m + cm * n;
        for &(m, n) in GRIDS {
            for &(mr, nr) in RATIOS {
                for nth in [2usize, 4, 8, 16] {
                    let (cm, cn, ..) = chunk_grid(m, n, mr, nr, nth);
                    let chunks = cm * cn;
                    // Only a band that fits along the axis is a real alternative:
                    // one clamped shorter would be a different chunk count, and a
                    // smaller chunk count trivially re-reads less.
                    if chunks > m || chunks > n {
                        continue;
                    }
                    let (m_ext, n_ext) = (m * mr, n * nr);
                    let ours = traffic(cm, cn, m_ext, n_ext);
                    let band_m = traffic(chunks, 1, m_ext, n_ext);
                    let band_n = traffic(1, chunks, m_ext, n_ext);
                    assert!(
                        ours <= band_m.min(band_n),
                        "{cm}x{cn} costs {ours}, bands cost {band_m}/{band_n} \
                         on {m}x{n} panels, {mr}x{nr} kernel, {nth} threads"
                    );
                }
            }
        }
    }
}

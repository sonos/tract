use super::{MatMatMul, Suitable};

fn order_f(a: f32, b: f32) -> std::cmp::Ordering {
    if a < b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
}

/// Analytic matmul-kernel cost model. Models each kernel's runtime as
/// `a * padded_work + b * n_tiles + c + restream * a_restream`, where `padded_work` is the
/// MAC count after rounding M and N up to the tile size, `n_tiles = ceil(m/mr) * ceil(n/nr)`,
/// and `a_restream = ceil(m/mr)*mr * ceil(n/nr)*k` is the packed-A re-stream volume (the
/// weight is read once per n-pass). `a` is the inverse steady-state throughput, `b` the
/// per-tile setup, `c` the fixed call overhead; these are fit per-kernel by least squares from
/// a `tract cost-model gather` dataset. `restream` is a single per-model coefficient for the
/// cost the per-kernel terms cannot express: a kernel is fit in isolation with its weight
/// cache-resident, but in a real model the weight is evicted between layers, so a small-`n`
/// kernel that re-streams a large `A` fewer times (wider `nr`) wins even when its isolated time
/// ties a narrower one. It is 0 both when un-calibrated and on cores whose last-level cache
/// cannot keep a packed `A` resident even in a warm loop: there `A` streams from memory in
/// isolation too, so no gap remains to correct and a cold calibration yields 0 — this is the
/// case for the small-cache LITTLE aarch64/arm32 cohorts, so their `0e0` is a measured no-op,
/// not a missing calibration.
///
/// For skinny-K (small K), an additional `skinny_k_penalty` term captures the overhead of
/// packing and kernel launch when K is small relative to the tile size. The penalty scales
/// inversely with K and is calibrated per-kernel.
#[derive(Debug)]
pub struct LinearCostModel<'a> {
    pub default_kernel: &'a str,
    pub kernels: &'a [&'a str],
    pub coeffs: &'a [[f32; 4]], // [a, b, c, skinny_k_penalty]
    pub restream: f32,
}

impl<'a> LinearCostModel<'a> {
    fn predicted(&self, ix: usize, m: usize, k: usize, n: usize, mr: usize, nr: usize) -> f32 {
        let (stream, tile, fixed) = self.terms(ix, m, k, n, mr, nr);
        stream + tile + fixed
    }

    /// Streaming work, per-tile work, and the part that does not shrink when the
    /// operator is split. The skinny-K penalty sits with the fixed term: it is a
    /// launch cost, and the coefficient is zero until a calibration fills it in.
    pub fn terms(
        &self,
        ix: usize,
        m: usize,
        k: usize,
        n: usize,
        mr: usize,
        nr: usize,
    ) -> (f32, f32, f32) {
        let coeffs = &self.coeffs[ix];
        let padded_work = (m.div_ceil(mr) * mr * n.div_ceil(nr) * nr * k) as f32;
        let n_tiles = (m.div_ceil(mr) * n.div_ceil(nr)) as f32;
        let a_restream = (m.div_ceil(mr) * mr * n.div_ceil(nr) * k) as f32;
        let tile_area = (mr * nr) as f32;
        let skinny_k_factor = if (k as f32) < tile_area { tile_area / k as f32 } else { 1.0 };
        let stream = coeffs[0] * padded_work + self.restream * a_restream;
        let tile = coeffs[1] * n_tiles;
        let fixed = coeffs[2] + coeffs[3] * (skinny_k_factor - 1.0);
        (stream, tile, fixed)
    }

    /// The fitted kernel this model predicts fastest among `suitable`, by name. The name comes
    /// from the model's own table rather than the list, so a tier can pass it on as its answer.
    ///
    /// One thread per kernel. [`Self::preferred_scaled`] is the same choice once a kernel's
    /// predicted time is divided by the threads that will actually run it; dividing every
    /// candidate by the same count leaves this answer unchanged.
    pub fn preferred(
        &self,
        suitable: &[Suitable],
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<&'a str> {
        if let (Some(m), Some(k), Some(n)) = (m, k, n) {
            return self.preferred_scaled(suitable, m, k, n, |_| 1);
        }
        // A dim the caller could not pin leaves the shape terms nothing to say, so all that is
        // left is the kernel the model was fitted around. Whether the suitable kernels include it
        // is the caller's business, and where they do not the generic rules answer instead.
        Some(self.default_kernel)
    }

    /// [`Self::preferred`] with each candidate's predicted time divided by the threads
    /// `threads_for` says that kernel will run on. A divisor below 1 counts as one.
    ///
    /// The divisor has to be constant across kernels that share a pipe. Giving one of them
    /// more threads than the pipe can absorb would crown it for a speedup it cannot deliver.
    pub fn preferred_scaled(
        &self,
        suitable: &[Suitable],
        m: usize,
        k: usize,
        n: usize,
        threads_for: impl Fn(&dyn MatMatMul) -> usize,
    ) -> Option<&'a str> {
        let best = suitable
            .iter()
            .filter_map(|(mmm, _, _)| {
                // nr==1 (matrix-vector) kernels are weighed only for the mmv path
                // (n==1). For n>=2 they are excluded, else a degenerate shape can be
                // handed a nr==1 kernel that pads N catastrophically.
                if mmm.nr() == 1 && n != 1 {
                    return None;
                }
                let ix = self.kernels.iter().position(|name| *name == mmm.name())?;
                let threads = threads_for(mmm.as_ref()).max(1) as f32;
                let t = self.predicted(ix, m, k, n, mmm.mr(), mmm.nr()) / threads;
                Some((t, self.kernels[ix]))
            })
            .min_by(|a, b| order_f(a.0, b.0))
            .map(|(_, name)| name);
        // No suitable kernel in the table: same fallback as [`Self::preferred`].
        best.or(Some(self.default_kernel))
    }

    /// Kernel whose parallel time is lowest. `roof_of` is the most workers that
    /// kernel may use; the caller decides that from the machine and from whether
    /// the streaming term dominates. One worker is the inline path and pays no
    /// entry cost, so this matches [`Self::preferred`] when every roof is 1.
    #[allow(clippy::too_many_arguments)]
    pub fn preferred_parallel(
        &self,
        suitable: &[Suitable],
        m: usize,
        k: usize,
        n: usize,
        roof_of: impl Fn(&dyn MatMatMul, bool) -> usize,
        panel_threshold: usize,
        entry: f32,
    ) -> Option<&'a str> {
        let best = suitable
            .iter()
            .filter_map(|(mmm, _, _)| {
                if mmm.nr() == 1 && n != 1 {
                    return None;
                }
                let ix = self.kernels.iter().position(|name| *name == mmm.name())?;
                let (stream, tile, fixed) = self.terms(ix, m, k, n, mmm.mr(), mmm.nr());
                let roof = roof_of(mmm.as_ref(), stream > tile + fixed).max(1);
                let panels = m.div_ceil(mmm.mr()) * n.div_ceil(mmm.nr());
                let (_workers, time) =
                    schedule_workers(stream, tile, fixed, roof, panels, panel_threshold, entry);
                Some((time, self.kernels[ix]))
            })
            .min_by(|a, b| order_f(a.0, b.0))
            .map(|(_, name)| name);
        best.or(Some(self.default_kernel))
    }
}

/// Seconds charged once for leaving the inline path.
///
/// Sits between the two costs of a short-K wide SME matmul on the M4 table
/// (tile term about 2.6e-5, ninety panels): two workers do not earn it, four
/// do. Not a thread-count table.
pub const PARALLEL_ENTRY_SECONDS: f32 = 2.0e-5;

/// Workers for one already-chosen kernel, and the predicted seconds at that count.
///
/// `stream` and `tile` both shrink with the worker count, up to `roof`. `fixed`
/// does not. A split pays `entry` once, so a cut that does not earn it stays at
/// one worker. Fewer panels than `panel_threshold` is the inline path.
pub fn schedule_workers(
    stream: f32,
    tile: f32,
    fixed: f32,
    roof: usize,
    panels: usize,
    panel_threshold: usize,
    entry: f32,
) -> (usize, f32) {
    let inline = stream + tile + fixed;
    if panels < panel_threshold || roof <= 1 {
        return (1, inline);
    }
    let roof = roof.min(panels).max(1);
    let variable = stream + tile;
    let mut best_t = 1usize;
    let mut best_time = inline;
    for t in 2..=roof {
        let time = variable / (t as f32) + fixed + entry;
        if time < best_time {
            best_time = time;
            best_t = t;
        }
    }
    (best_t, best_time)
}

#[cfg(test)]
mod schedule_tests {
    use super::{PARALLEL_ENTRY_SECONDS, schedule_workers};

    /// A short-K wide SME matmul on the M4 table: the per-tile term dominates,
    /// two workers do not earn the entry cost, four do.
    #[test]
    fn short_k_stays_inline_until_the_split_pays() {
        let stream = 6.05e-6;
        let tile = 2.644e-5;
        let fixed = 3.986e-6;
        let (t2, _) = schedule_workers(stream, tile, fixed, 2, 90, 64, PARALLEL_ENTRY_SECONDS);
        let (t4, _) = schedule_workers(stream, tile, fixed, 4, 90, 64, PARALLEL_ENTRY_SECONDS);
        assert_eq!(t2, 1);
        assert_eq!(t4, 4);
    }

    #[test]
    fn a_handful_of_panels_never_splits() {
        let (t, _) = schedule_workers(1.0, 1.0, 0.0, 8, 4, 64, 0.0);
        assert_eq!(t, 1);
    }

    /// A pipe-filling GEMM is scored with the pipe count as its roof. One pipe
    /// stays inline. Three pipes take three workers, because each pipe pays
    /// the entry once and the streaming term shrinks.
    #[test]
    fn a_pipe_roof_is_taken_in_full() {
        let stream = 1.836e-4;
        let tile = 7.52e-5;
        let fixed = 3.986e-6;
        let (t1, _) = schedule_workers(stream, tile, fixed, 1, 256, 64, PARALLEL_ENTRY_SECONDS);
        let (t3, _) = schedule_workers(stream, tile, fixed, 3, 256, 64, PARALLEL_ENTRY_SECONDS);
        assert_eq!(t1, 1);
        assert_eq!(t3, 3);
    }
}

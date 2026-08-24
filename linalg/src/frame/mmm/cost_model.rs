use super::Suitable;

fn order_f(a: f32, b: f32) -> std::cmp::Ordering {
    if a < b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
}

/// Analytic matmul-kernel cost model. Models each kernel's runtime as
/// `a * padded_work + b * n_tiles + c + restream * a_restream`, where `padded_work` is the
/// MAC count after rounding M and N up to the tile size, `n_tiles = ceil(m/mr) * ceil(n/nr)`,
/// and `a_restream = ceil(m/mr)*mr * ceil(n/nr) * k` is the packed-A re-stream volume (the
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
/// not a missing calibration. `preferred` returns the argmin over the suitable kernels it is given.
#[derive(Debug)]
pub struct LinearCostModel<'a> {
    pub default_kernel: &'a str,
    pub kernels: &'a [&'a str],
    pub coeffs: &'a [[f32; 3]],
    pub restream: f32,
}

impl<'a> LinearCostModel<'a> {
    fn predicted(&self, ix: usize, m: usize, k: usize, n: usize, mr: usize, nr: usize) -> f32 {
        let coeffs = &self.coeffs[ix];
        let padded_work = (m.div_ceil(mr) * mr * n.div_ceil(nr) * nr * k) as f32;
        let n_tiles = (m.div_ceil(mr) * n.div_ceil(nr)) as f32;
        let a_restream = (m.div_ceil(mr) * mr * n.div_ceil(nr) * k) as f32;
        coeffs[0] * padded_work + coeffs[1] * n_tiles + coeffs[2] + self.restream * a_restream
    }

    /// The fitted kernel this model predicts fastest among `suitable`, by name. The name comes
    /// from the model's own table rather than the list, so a tier can pass it on as its answer.
    pub fn preferred(
        &self,
        suitable: &[Suitable],
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<&'a str> {
        if let (Some(m), Some(k), Some(n)) = (m, k, n) {
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
                    let t = self.predicted(ix, m, k, n, mmm.mr(), mmm.nr());
                    Some((t, self.kernels[ix]))
                })
                .min_by(|a, b| order_f(a.0, b.0))
                .map(|(_, name)| name);
            if best.is_some() {
                return best;
            }
        }
        // A dim the caller could not pin leaves the shape terms nothing to say, so all that is
        // left is the kernel the model was fitted around. Whether the suitable kernels include it
        // is the caller's business, and where they do not the portable rules answer instead.
        Some(self.default_kernel)
    }
}

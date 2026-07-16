use super::MatMatMul;

fn order_f(a: f32, b: f32) -> std::cmp::Ordering {
    if a < b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
}

/// Analytic matmul-kernel cost model. Models each kernel's runtime as
/// `a * padded_work + b * n_tiles + c + restream * a_restream`, where `padded_work` is the
/// MAC count after rounding M and N up to the tile size, `n_tiles = ceil(m/mr) * ceil(n/nr)`,
/// and `a_restream = ceil(m/mr)*mr * ceil(n/nr) * k` is the packed-A re-stream volume (the
/// weight is read once per n-pass). `a` is the inverse steady-state throughput, `b` the
/// per-tile setup, `c` the fixed call overhead; these are fit per-kernel by least squares from
/// a `tract cost-model gather` dataset. `restream` is a single per-model coefficient (0 when
/// un-calibrated) for the cost the per-kernel terms cannot express: a kernel is fit in
/// isolation with its weight cache-resident, but in a real model the weight is evicted between
/// layers, so a small-`n` kernel that re-streams a large `A` fewer times (wider `nr`) wins
/// even when its isolated time ties a narrower one. `pick` returns the argmin over the impls.
#[derive(Debug)]
pub struct LinearCostModel<'a> {
    pub default_kernel: &'a str,
    pub kernels: &'a [&'a str],
    pub coeffs: &'a [[f32; 3]],
    pub restream: f32,
}

impl LinearCostModel<'_> {
    fn predicted(&self, ix: usize, m: usize, k: usize, n: usize, mr: usize, nr: usize) -> f32 {
        let coeffs = &self.coeffs[ix];
        let padded_work = (m.div_ceil(mr) * mr * n.div_ceil(nr) * nr * k) as f32;
        let n_tiles = (m.div_ceil(mr) * n.div_ceil(nr)) as f32;
        let a_restream = (m.div_ceil(mr) * mr * n.div_ceil(nr) * k) as f32;
        coeffs[0] * padded_work + coeffs[1] * n_tiles + coeffs[2] + self.restream * a_restream
    }

    pub fn pick(
        &self,
        impls: &[Box<dyn MatMatMul>],
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Box<dyn MatMatMul> {
        if let (Some(m), Some(k), Some(n)) = (m, k, n) {
            let best = impls
                .iter()
                .filter_map(|imp| {
                    // Matrix-vector kernels (nr==1) are the mmv path; never a mmm candidate.
                    // Without this a degenerate shape can be handed a nr==1 kernel.
                    if imp.nr() == 1 {
                        return None;
                    }
                    let ix = self.kernels.iter().position(|name| *name == imp.name())?;
                    let t = self.predicted(ix, m, k, n, imp.mr(), imp.nr());
                    Some((t, imp))
                })
                .min_by(|a, b| order_f(a.0, b.0))
                .map(|(_, imp)| imp.clone());
            if let Some(best) = best {
                return best;
            }
        }
        impls.iter().find(|k| k.name() == self.default_kernel).unwrap().clone()
    }
}

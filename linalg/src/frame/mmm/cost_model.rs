use super::MatMatMul;

fn order_f(a: f32, b: f32) -> std::cmp::Ordering {
    if a < b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater }
}

/// Analytic matmul-kernel cost model. Models each kernel's runtime as
/// `a * padded_work + b * n_tiles + c`, where `padded_work` is the MAC count after
/// rounding M and N up to the tile size, `n_tiles = ceil(m/mr) * ceil(n/nr)`, and the
/// per-kernel coefficients are fit by least squares from a `tract cost-model gather`
/// dataset. `a` is the inverse steady-state throughput, `b` the per-tile setup, `c` the
/// fixed call overhead. `pick` returns the argmin over the candidate impls.
#[derive(Debug)]
pub struct LinearCostModel<'a> {
    pub default_kernel: &'a str,
    pub kernels: &'a [&'a str],
    pub coeffs: &'a [[f32; 3]],
}

impl LinearCostModel<'_> {
    fn predicted(coeffs: &[f32; 3], m: usize, k: usize, n: usize, mr: usize, nr: usize) -> f32 {
        let padded_work = (m.div_ceil(mr) * mr * n.div_ceil(nr) * nr * k) as f32;
        let n_tiles = (m.div_ceil(mr) * n.div_ceil(nr)) as f32;
        coeffs[0] * padded_work + coeffs[1] * n_tiles + coeffs[2]
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
                    let ix = self.kernels.iter().position(|name| *name == imp.name())?;
                    let t = Self::predicted(&self.coeffs[ix], m, k, n, imp.mr(), imp.nr());
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

use tract_data::internal::*;
use tract_data::itertools::{izip, Itertools};

use super::MatMatMul;

fn order_f<F: tract_num_traits::Float>(&a: &F, &b: &F) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else {
        std::cmp::Ordering::Greater
    }
}

#[derive(Debug)]
pub struct CostModel<'a> {
    pub big_product_mkn_threshold: f32,
    pub big_product_kernel_choice: &'a str,
    pub kernels: &'a [&'a str],
    pub mrs: &'a [u32],
    pub nrs: &'a [u32],
    pub feat_norm_mean: &'a [f32],
    pub feat_norm_stddev: &'a [f32],
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub b2: &'a [f32],
}

impl CostModel<'_> {
    pub fn features(&self, m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut feat = vec![
            (m as f32).ln(),
            (k as f32).ln(),
            (n as f32).ln(),
            (n as f32 * m as f32 * k as f32).ln(),
        ];
        for &mr in self.mrs {
            let mr = mr as usize;
            feat.push((m % mr) as f32);
            feat.push((m % mr != 0) as usize as f32);
        }
        for &nr in self.nrs {
            let nr = nr as usize;
            feat.push((n % nr) as f32);
            feat.push((n % nr != 0) as usize as f32);
        }
        feat
    }

    fn normalize(&self, feat: &mut [f32]) {
        izip!(feat, self.feat_norm_mean, self.feat_norm_stddev)
            .for_each(|(x, m, s)| *x = (*x - m) / s)
    }

    fn dnn(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        let x = tract_ndarray::Array1::from_vec(x.to_vec());
        let w = tract_ndarray::Array2::from_shape_vec([b.len(), x.len()], w.to_vec()).unwrap();
        let b = tract_ndarray::Array1::from_vec(b.to_vec());
        (w.dot(&x) + b).to_vec()
    }

    pub fn predict(&self, m: usize, k: usize, n: usize) -> &str {
        let mut x = self.features(m, k, n);
        self.normalize(&mut x);
        let mut hidden = Self::dnn(&x, self.w1, self.b1);
        (crate::generic().tanh_f32)().run(&mut hidden).unwrap();
        let output = Self::dnn(&hidden, self.w2, self.b2);
        let ix = output.iter().copied().position_max_by(order_f).unwrap();
        self.kernels[ix]
    }

    pub fn pick(
        &self,
        impls: &[Box<dyn MatMatMul>],
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Box<dyn MatMatMul> {
        if let (Some(m), Some(k), Some(n)) = (m, k, n) {
            let choice = self.predict(m, k, n);
            impls.iter().find(|k| k.name() == choice).unwrap().clone()
        } else {
            impls.iter().find(|k| k.name() == self.big_product_kernel_choice).unwrap().clone()
        }
    }
}

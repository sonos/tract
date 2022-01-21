use std::fmt;

use tract_data::internal::DimLike;

#[derive(Debug, Clone)]
pub struct CostModel {
    pub mr: usize,
    pub nr: usize,
    pub intercept: f64,
    pub coef: Vec<f64>,
}

impl CostModel {
    pub fn features(
        mr: usize,
        nr: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> impl IntoIterator<Item = f64> + fmt::Debug {
        let rows = m.divceil(mr);
        let cols = n.divceil(nr);
        let rows_recip = (rows as f64 + 1.0).recip();
        let cols_recip = (cols as f64 + 1.0).recip();
        vec![
            (rows * cols) as f64,
            (rows * cols * k) as f64,
            rows_recip,
            rows_recip * k as f64,
            cols_recip,
            cols_recip * k as f64,
            (cols * ((m % mr) != 0) as usize) as f64,
            (rows * ((n % nr) != 0) as usize) as f64,
            (cols * (m % mr) as usize) as f64,
            (rows * (n % nr) as usize) as f64,
            (((n % nr) != 0) as usize * ((m % mr) != 0) as usize) as f64,
        ]
    }

    pub fn predict(&self, m: usize, k: usize, n: usize) -> f64 {
        let feats = Self::features(self.mr, self.nr, m, k, n);
        self.intercept + self.coef.iter().zip(feats).map(|(c, x)| c * x).sum::<f64>()
    }
}

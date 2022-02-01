use std::fmt;

use tract_data::internal::DimLike;

#[derive(Debug)]
pub struct CostModel {
    pub mr: usize,
    pub nr: usize,
    pub alpha: f32,
    pub intercept: f32,
    pub forest: randomforest::RandomForestRegressor,
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
            (rows * cols * k) as f64, // keep me first !
            (rows * cols) as f64,
            rows_recip,
            rows_recip * k as f64,
            cols_recip,
            cols_recip * k as f64,
            (cols * (m % mr) as usize) as f64,
            (rows * (n % nr) as usize) as f64,
        ]
    }

    pub fn predict(&self, m: usize, k: usize, n: usize) -> f32 {
        let rows = m.divceil(self.mr);
        let cols = n.divceil(self.nr);
        let feats: Vec<f64> = Self::features(self.mr, self.nr, m, k, n).into_iter().collect();
        self.intercept
            + (self.alpha * (rows * cols * k) as f32 + self.forest.predict(&*feats) as f32)
    }
}

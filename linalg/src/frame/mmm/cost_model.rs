use std::fmt;

use tract_data::internal::DimLike;

pub struct Model {
    pub mr: usize,
    pub nr: usize,
    pub intercept: f64,
    pub coef: Vec<f64>,
}

impl Model {
    pub fn features(
        mr: usize,
        nr: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> impl IntoIterator<Item = f64> + fmt::Debug {
        let rows = m.divceil(mr);
        let cols = n.divceil(nr);
        vec![
            /*
            k as f64,
            (k * k) as f64,
            */
            (rows * cols) as f64,
            (rows * cols * k) as f64,
            (rows * rows * cols * cols) as f64,
            (rows * rows * cols * cols * k) as f64,
            /*
            cols as f64,
            rows as f64,
            */
            /*
            (rows * rows) as f64,
            (cols * cols) as f64,
            ((m == 1) as usize) as f64,
            ((rows == 1) as usize) as f64,
            ((cols == 1) as usize) as f64,
            ((rows == 1) as usize * cols) as f64,
            ((cols == 1) as usize * rows) as f64,
            (rows * rows * cols) as f64,
            (rows * cols * cols) as f64,
            */
            (cols * ((m % mr) != 0) as usize) as f64,
            (rows * ((n % nr) != 0) as usize) as f64,
            (cols * (m % mr) as usize) as f64,
            (rows * (n % nr) as usize) as f64,
            /*
            (cols * rows * (m % mr) as usize) as f64,
            (cols * rows * (n % nr) as usize) as f64,
            (cols * cols * (m % mr) as usize) as f64,
            (rows * rows * (n % nr) as usize) as f64,
            (cols * rows * ((n % nr) != 0) as usize * ((m % mr) != 0) as usize) as f64,
            */
            (((n % nr) != 0) as usize * ((m % mr) != 0) as usize) as f64,
            /*
            ((n % nr) * (m % mr)) as f64,
            (cols * cols * rows * rows * ((n % nr) != 0) as usize * ((m % mr) != 0) as usize)
                as f64,
            */
        ]
    }

    pub fn predict(&self, m: usize, k: usize, n: usize) -> f64 {
        let feats = Self::features(self.mr, self.nr, m, k, n);
        self.intercept + self.coef.iter().zip(feats).map(|(c, x)| c * x).sum::<f64>()
    }
}

use std::fmt;
use std::ops::{Add, Deref, Mul};

use num_traits::Zero;

use super::*;

pub trait QMatMatMul<TA, TB, TC, TI>:
    fmt::Debug + fmt::Display + objekt::Clone + Send + Sync
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
{
    fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI>;
    fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI>;

    unsafe fn set_zero_point_a_scalar(&mut self, value: TI);
    unsafe fn set_zero_point_a_vector(&mut self, values: Vec<TI>);
    unsafe fn set_zero_point_b_scalar(&mut self, value: TI);
    unsafe fn set_zero_point_b_vector(&mut self, values: Vec<TI>);
}

clone_trait_object!(<TA, TB, TC, TI> QMatMatMul<TA, TB, TC, TI> where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
);

#[derive(Debug, Clone)]
pub struct QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    pub mmm: MatMatMulImpl<K, TA, TB, TC, TI>,
    pub zero_point_a: Option<Vec<TI>>,
    pub zero_point_b: Option<Vec<TI>>,
}

impl<K, TA, TB, TC, TI> From<MatMatMulImpl<K, TA, TB, TC, TI>> for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn from(mmm: MatMatMulImpl<K, TA, TB, TC, TI>) -> QMatMatMulImpl<K, TA, TB, TC, TI> {
        QMatMatMulImpl { mmm, zero_point_a: None, zero_point_b: None }
    }
}

impl<K, TA, TB, TC, TI> Deref for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    type Target = MatMatMulImpl<K, TA, TB, TC, TI>;
    fn deref(&self) -> &Self::Target {
        &self.mmm
    }
}

unsafe impl<K, TA, TB, TC, TI> Send for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
}

unsafe impl<K, TA, TB, TC, TI> Sync for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
}

impl<K, TA, TB, TC, TI> QMatMatMul<TA, TB, TC, TI> for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero + fmt::Debug,
    TB: Copy + Zero + fmt::Debug,
    TC: Copy + fmt::Debug,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI> {
        &self.mmm
    }
    fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI> {
        &mut self.mmm
    }
    unsafe fn set_zero_point_a_scalar(&mut self, value: TI) {
        self.zero_point_a = Some(vec![value; self.m() + K::mr() - 1 / K::mr() * K::mr()])
    }

    unsafe fn set_zero_point_b_scalar(&mut self, value: TI) {
        self.zero_point_b = Some(vec![value; self.n() + K::nr() - 1 / K::nr() * K::nr()])
    }

    unsafe fn set_zero_point_a_vector(&mut self, mut values: Vec<TI>) {
        let wanted = self.m() + K::mr() - 1 / K::mr() * K::mr();
        while values.len() < wanted {
            values.push(values[values.len() - 1])
        }
        self.zero_point_a = Some(values)
    }

    unsafe fn set_zero_point_b_vector(&mut self, mut values: Vec<TI>) {
        let wanted = self.n() + K::nr() - 1 / K::nr() * K::nr();
        while values.len() < wanted {
            values.push(values[values.len() - 1])
        }
        self.zero_point_b = Some(values)
    }
}

impl<K, TA, TB, TC, TI> fmt::Display for QMatMatMulImpl<K, TA, TB, TC, TI>
where
    TA: Copy + Zero,
    TB: Copy + Zero,
    TC: Copy,
    TI: Copy + Add + Mul + Zero + fmt::Debug,
    K: MatMatMulKer<TA, TB, TC, TI>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "A:{}, B:{} C:{} (m:{}, k:{}, n:{})",
            self.a_storage, self.b_storage, self.c_storage, self.m, self.k, self.n
        )
    }
}

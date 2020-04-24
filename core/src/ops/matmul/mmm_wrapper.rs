use num_traits::Zero;
use std::fmt;
use std::ops::{Add, Deref, Mul};

use crate::internal::*;
use crate::ops::quant::QParams;

use tract_linalg::mmm::{FusedSpec, MatMatMul, QMatMatMul};

#[derive(Clone, Debug, Educe)]
#[educe(Hash)]
pub enum MMMWrapper<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    Plain(Box<dyn MatMatMul<TA, TB, TC, TI>>),
    Quant(Box<dyn QMatMatMul<TA, TB, TC, TI>>),
}

impl<TA, TB, TC, TI> MMMWrapper<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    pub fn as_mmm(&self) -> &dyn MatMatMul<TA, TB, TC, TI> {
        match self {
            MMMWrapper::Plain(a) => a.as_ref(),
            MMMWrapper::Quant(a) => a.as_mmm(),
        }
    }

    pub fn as_mmm_mut(&mut self) -> &mut dyn MatMatMul<TA, TB, TC, TI> {
        match self {
            MMMWrapper::Plain(a) => a.as_mut(),
            MMMWrapper::Quant(a) => a.as_mmm_mut(),
        }
    }

    pub fn as_quant(&self) -> Option<&dyn QMatMatMul<TA, TB, TC, TI>> {
        match self {
            MMMWrapper::Plain(_) => None,
            MMMWrapper::Quant(a) => Some(a.deref()),
        }
    }

    pub fn as_quant_mut(&mut self) -> Option<&mut dyn QMatMatMul<TA, TB, TC, TI>> {
        match self {
            MMMWrapper::Plain(_) => None,
            MMMWrapper::Quant(ref mut a) => Some(a.as_mut()),
        }
    }

    pub unsafe fn run(&self, a: *const TA, b: *const TB, c: *mut TC, non_linear: &[FusedSpec<TI>]) {
        match self {
            MMMWrapper::Plain(p) => p.run(a, b, c, non_linear),
            MMMWrapper::Quant(q) => q.run(a, b, c, non_linear),
        }
    }

    pub fn set_quant_params(&mut self, params: &QParams) -> TractResult<()> {
        let q = self.as_quant_mut().ok_or("try to zero_point on a float mat mul")?;
        unsafe {
            if let Some(t) = params.zero_point_a.as_ref() {
                if t.rank() == 0 {
                    q.set_zero_point_a_scalar(*t.to_scalar()?)
                } else {
                    q.set_zero_point_a_vector(t.as_slice()?.to_vec())
                }
            }
            if let Some(t) = params.zero_point_b.as_ref() {
                if t.rank() == 0 {
                    q.set_zero_point_b_scalar(*t.to_scalar()?)
                } else {
                    q.set_zero_point_b_vector(t.as_slice()?.to_vec())
                }
            }
            if let Some(t) = params.zero_point_c.as_ref() {
                q.set_zero_point_c_scalar(t.cast_to_scalar()?)
            }
            if let Some(factor) = params.scale_factor {
                q.set_scale_factor(factor);
            }
        }
        Ok(())
    }
}

impl<TA, TB, TC, TI> fmt::Display for MMMWrapper<TA, TB, TC, TI>
where
    TA: Datum + Copy + Zero,
    TB: Datum + Copy + Zero,
    TC: Datum + Copy,
    TI: Datum + Copy + Add + Mul + Zero + fmt::Debug,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MMMWrapper::Plain(a) => write!(fmt, "{}", a),
            MMMWrapper::Quant(a) => write!(fmt, "{}", a),
        }
    }
}

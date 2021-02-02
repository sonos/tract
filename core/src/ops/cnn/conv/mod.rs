mod depth_wise;
mod im2col;
#[cfg(test)]
pub mod proptest;
#[cfg(test)]
mod proptest_q;
mod q_sum_b;
mod unary;

use crate::internal::*;

pub use self::im2col::Im2Col;
pub(crate) use self::q_sum_b::QSumB;
pub use self::unary::ConvUnary;

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum KernelFormat {
    OIHW,
    HWIO,
}

impl Default for KernelFormat {
    fn default() -> KernelFormat {
        KernelFormat::OIHW
    }
}

impl KernelFormat {
    pub fn h_axis(&self) -> usize {
        match self {
            KernelFormat::OIHW => 2,
            KernelFormat::HWIO => 0,
        }
    }

    pub fn spatial_shape<'a, D: DimLike>(&self, full_shape: &'a [D]) -> &'a [D] {
        &full_shape[self.h_axis()..][..full_shape.len() - 2]
    }

    pub fn i<'a, D: DimLike>(&self, full_shape: &'a [D]) -> &'a D {
        match self {
            KernelFormat::OIHW => &full_shape[1],
            KernelFormat::HWIO => &full_shape[full_shape.len() - 2],
        }
    }

    pub fn o<'a, D: DimLike>(&self, full_shape: &'a [D]) -> &'a D {
        match self {
            KernelFormat::OIHW => &full_shape[0],
            KernelFormat::HWIO => &full_shape[full_shape.len() - 1],
        }
    }
}

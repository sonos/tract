mod depth_wise;
mod im2col;
#[cfg(test)]
mod proptest;
mod unary;
mod q_sum_b;

pub use self::im2col::Im2Col;
pub use self::unary::ConvUnary;
pub(crate) use self::q_sum_b::QSumB;

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
}

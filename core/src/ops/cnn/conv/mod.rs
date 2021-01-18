mod depth_wise;
mod im2col;
#[cfg(test)]
mod proptest;
mod q_sum_b;
mod unary;

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
}

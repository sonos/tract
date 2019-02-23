mod conv_gemm;
mod direct;
mod gen;
mod im2col;
mod unary;

pub use self::direct::Direct;
pub use self::gen::Conv;
pub use self::unary::ConvUnary;

#[derive(Debug, Copy, Clone, PartialEq)]
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
    pub(super) fn h_axis(&self) -> usize {
        match self {
            KernelFormat::OIHW => 2,
            KernelFormat::HWIO => 0,
        }
    }
}

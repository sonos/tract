mod depth_wise;
mod direct;
mod gen;
mod im2col;
mod mat_mat;
mod unary;
mod vec_mat;

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

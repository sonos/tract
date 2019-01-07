extern crate lazy_static;
extern crate log;
extern crate num;

pub mod f16;
pub mod frame;
mod generic;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86_64_fma;

pub use self::frame::{MatMul, PackedMatMul};

pub struct Ops {
    pub smm: Box<Fn(usize,usize,usize) -> Box<MatMul<f32>> + Send + Sync>,
    pub dmm: Box<Fn(usize,usize,usize) -> Box<MatMul<f64>> + Send + Sync>,
}

pub fn generic() -> Ops {
    Ops {
        smm: Box::new(|m,k,n| Box::new(PackedMatMul::<generic::SMatMul, f32>::new(m,k,n))),
        dmm: Box::new(|m,k,n| Box::new(PackedMatMul::<generic::DMatMul, f64>::new(m,k,n))),
    }
}

lazy_static::lazy_static! {
    static ref OPS: Ops = { generic() };
}

pub fn ops() -> &'static Ops {
    &*OPS
}

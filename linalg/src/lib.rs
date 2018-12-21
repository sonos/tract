#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate num;

pub mod f16;
pub mod frame;
mod generic;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86_64_fma;

pub use self::frame::MatMul;

pub struct Ops {
    pub smm: Box<frame::MatMul<f32>>,
    pub dmm: Box<frame::MatMul<f64>>,
}

pub fn generic() -> Ops {
    Ops {
        smm: Box::new(generic::SMatMul),
        dmm: Box::new(generic::DMatMul),
    }
}


lazy_static! {
    static ref OPS: Ops = {
        generic()
    };
}

pub fn ops() -> &'static Ops {
    &*OPS
}

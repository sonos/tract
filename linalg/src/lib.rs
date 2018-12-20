#[macro_use]
extern crate log;

mod frame;
mod generic;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86_64_fma;

/*
struct Ops {
    smm: Box<frame::MatMul>,
}

pub fn generic() -> Ops {
    Ops {
        smm: 
    }
}
*/

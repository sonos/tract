#[macro_use]
extern crate derive_new;
extern crate lazy_static;
extern crate libc;
extern crate log;
#[macro_use]
extern crate objekt;
extern crate num_traits;
#[cfg(test)]
extern crate proptest;

pub mod align;
pub mod f16;
#[macro_use]
pub mod frame;
mod generic;

#[cfg(target_arch = "x86_64")]
pub mod x86_64_fma;

#[cfg(target_arch = "aarch64")]
pub mod arm64;

#[cfg(any(target_arch = "arm", target_arch = "armv7"))]
pub mod arm32;

pub use self::frame::*;

pub struct Ops {
    pub svmm: Box<dyn Fn(usize, usize) -> Box<dyn VecMatMul<f32>> + Send + Sync>,
    pub stile: Box<dyn Fn(usize, usize, usize) -> Box<dyn Tile<f32>> + Send + Sync>,
}

pub fn generic() -> Ops {
    Ops {
        svmm: Box::new(|k, n| Box::new(PackedVecMatMul::<generic::SVecMatMul8, f32>::new(k, n))),
        stile: Box::new(|m, k, n| Box::new(TileOp::<generic::STiling4x4, f32>::new(m, k, n))),
    }
}

#[allow(unreachable_code, unused_mut)]
pub fn best() -> Ops {
    let mut ops = generic();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            ops.stile = Box::new(|m, k, n| {
                Box::new(TileOp::<x86_64_fma::tile::STile16x6, f32>::new(m, k, n))
            });
            log::info!("x86_64/fma activated");
        }
    }
    #[cfg(any(target_arch = "arm", target_arch = "armv7"))]
    arm32::plug(&mut ops);
    #[cfg(target_arch = "aarch64")]
    arm64::plug(&mut ops);
    return ops;
}

lazy_static::lazy_static! {
    static ref OPS: Ops = {
        best()
    };
}

pub fn ops() -> &'static Ops {
    &*OPS
}

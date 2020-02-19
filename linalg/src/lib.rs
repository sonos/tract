#[macro_use]
extern crate derive_new;
extern crate lazy_static;
extern crate libc;
extern crate log;
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

pub use self::frame::lut;
pub use self::frame::mmm;
pub use self::frame::sigmoid;
pub use self::frame::tanh;

pub struct Ops {
    pub smmm: Box<
        dyn Fn(usize, usize, usize) -> Box<dyn mmm::MatMatMul<f32, f32, f32, f32>> + Send + Sync,
    >,
    pub qmmm_i8_i32: Box<
        dyn Fn(usize, usize, usize) -> Box<dyn mmm::QMatMatMul<i8, i8, i32, i32>> + Send + Sync,
    >,
    pub qmmm_u8_i32: Box<
        dyn Fn(usize, usize, usize) -> Box<dyn mmm::QMatMatMul<u8, u8, i32, i32>> + Send + Sync,
    >,
    pub qmmm_u8_u8:
        Box<dyn Fn(usize, usize, usize) -> Box<dyn mmm::QMatMatMul<u8, u8, u8, i32>> + Send + Sync>,
    pub qmmm_i8_i8:
        Box<dyn Fn(usize, usize, usize) -> Box<dyn mmm::QMatMatMul<i8, i8, i8, i32>> + Send + Sync>,
    pub ssigmoid: Box<dyn Fn() -> Box<dyn sigmoid::Sigmoid<f32>> + Send + Sync>,
    pub stanh: Box<dyn Fn() -> Box<dyn tanh::Tanh<f32>> + Send + Sync>,
    pub lut_u8: Box<dyn Fn(&[u8]) -> Box<dyn lut::Lut> + Send + Sync>,
}

pub fn generic() -> Ops {
    Ops {
        smmm: Box::new(|m, k, n| {
            Box::new(mmm::MatMatMulImpl::<
                generic::GenericMmm4x4<f32, f32, f32, f32>,
                f32,
                f32,
                f32,
                f32,
            >::new(m, k, n))
        }),
        qmmm_i8_i32: Box::new(|m, k, n| {
            Box::new(mmm::QMatMatMulImpl::from(mmm::MatMatMulImpl::<
                generic::GenericMmm4x4<i8, i8, i32, i32>,
                i8,
                i8,
                i32,
                i32,
            >::new(m, k, n)))
        }),
        qmmm_u8_i32: Box::new(|m, k, n| {
            Box::new(mmm::QMatMatMulImpl::from(mmm::MatMatMulImpl::<
                generic::GenericMmm4x4<u8, u8, i32, i32>,
                u8,
                u8,
                i32,
                i32,
            >::new(m, k, n)))
        }),
        qmmm_u8_u8: Box::new(|m, k, n| {
            Box::new(mmm::QMatMatMulImpl::from(mmm::MatMatMulImpl::<
                generic::GenericMmm4x4<u8, u8, u8, i32>,
                u8,
                u8,
                u8,
                i32,
            >::new(m, k, n)))
        }),
        qmmm_i8_i8: Box::new(|m, k, n| {
            Box::new(mmm::QMatMatMulImpl::from(mmm::MatMatMulImpl::<
                generic::GenericMmm4x4<i8, i8, i8, i32>,
                i8,
                i8,
                i8,
                i32,
            >::new(m, k, n)))
        }),
        ssigmoid: Box::new(|| Box::new(sigmoid::SigmoidImpl::<generic::SSigmoid4, f32>::new())),
        stanh: Box::new(|| Box::new(tanh::TanhImpl::<generic::STanh4, f32>::new())),
        lut_u8: Box::new(|table: &[u8]| Box::new(lut::LutImpl::<generic::GenericLut8>::new(table))),
    }
}

#[allow(unreachable_code, unused_mut)]
pub fn best() -> Ops {
    let mut ops = generic();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            ops.smmm = Box::new(|m, k, n| {
                Box::new(
                    mmm::MatMatMulImpl::<x86_64_fma::mmm::MatMatMulF32x16x6, f32, f32, f32, f32>::new(
                        m, k, n,
                    ),
                )
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

#[cfg(test)]
pub(crate) fn check_close(
    found: &[f32],
    expected: &[f32],
) -> proptest::test_runner::TestCaseResult {
    proptest::prop_assert!(
        found.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 0.001),
        "found: {:?} expected: {:?}",
        found,
        expected
    );
    Ok(())
}

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::redundant_closure_call)]
#![allow(clippy::len_zero)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(unexpected_cfgs)]
#![allow(unused_macros)]
#[macro_use]
extern crate derive_new;
extern crate lazy_static;
extern crate log;
extern crate num_traits;
#[macro_use]
extern crate paste;
#[cfg(test)]
extern crate proptest;

include!(concat!(env!("OUT_DIR"), "/extern_kernel_macro.rs"));

#[macro_use]
pub mod frame;
pub mod generic;
pub mod multithread;
use frame::element_wise::ElementWiseKer;
use frame::reduce::{MapReduceKer, ReduceKer};
use frame::unicast::UnicastKer;
use frame::{reduce, unicast, MatMatMul};
pub use generic::{ScaleShiftAndRound, Scaler};
#[cfg(target_arch = "x86_64")]
pub mod x86_64_fma;

#[cfg(target_arch = "aarch64")]
pub mod arm64;

#[cfg(any(target_arch = "arm", target_arch = "armv7", target_arch = "arm"))]
pub mod arm32;

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
pub mod wasm;

pub use self::frame::{element_wise, lut, mmm};

use tract_data::prelude::*;

pub type MMMImpl = Box<
    dyn Fn(Option<usize>, Option<usize>, Option<usize>) -> Box<dyn mmm::MatMatMul> + Send + Sync,
>;

type MMVImpl = Box<dyn Fn(Option<usize>, Option<usize>) -> Box<dyn mmm::MatMatMul> + Send + Sync>;

#[allow(clippy::type_complexity)]
pub struct Ops {
    mmm_impls: Vec<Box<dyn MatMatMul>>,

    mmm_f64: MMMImpl,
    mmv_f64: MMVImpl,

    mmm_f32: MMMImpl,
    mmv_f32: MMVImpl,

    mmm_f16: MMMImpl,
    mmv_f16: MMVImpl,

    qmmm_i32: MMMImpl,
    qmmv_i32: MMVImpl,

    pub leaky_relu_f16: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16, f16>> + Send + Sync>,
    pub leaky_relu_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32, f32>> + Send + Sync>,
    pub mul_by_scalar_f32:
        Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32, f32>> + Send + Sync>,
    pub mul_by_scalar_f16:
        Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16, f16>> + Send + Sync>,

    pub sigmoid_f16: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16>> + Send + Sync>,
    pub sigmoid_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub tanh_f16: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16>> + Send + Sync>,
    pub tanh_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub erf_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub lut_u8: Box<dyn Fn(&[u8]) -> Box<dyn lut::Lut> + Send + Sync>,

    pub max_f16: Box<dyn Fn() -> Box<dyn reduce::Reduce<f16>> + Send + Sync>,
    pub max_f32: Box<dyn Fn() -> Box<dyn reduce::Reduce<f32>> + Send + Sync>,

    pub sum_f16: Box<dyn Fn() -> Box<dyn reduce::Reduce<f16>> + Send + Sync>,
    pub sum_f32: Box<dyn Fn() -> Box<dyn reduce::Reduce<f32>> + Send + Sync>,

    pub unicast_mul_f16: Box<dyn Fn() -> Box<dyn unicast::Unicast<f16>> + Send + Sync>,
    pub unicast_mul_f32: Box<dyn Fn() -> Box<dyn unicast::Unicast<f32>> + Send + Sync>,

    pub softmax2_fastcompact_f16:
        Box<dyn Fn() -> Box<dyn reduce::MapReduce<f16, f16>> + Send + Sync>,
    pub softmax2_fastcompact_f32:
        Box<dyn Fn() -> Box<dyn reduce::MapReduce<f32, f32>> + Send + Sync>,
}

impl Ops {
    pub fn mmm_impls(&self) -> &[Box<dyn MatMatMul>] {
        &self.mmm_impls
    }

    pub fn mmm(
        &self,
        accumulator: DatumType,
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<Box<dyn mmm::MatMatMul>> {
        use DatumType::*;
        match accumulator {
            F64 => Some(if n == Some(1) { (self.mmv_f64)(m, k) } else { (self.mmm_f64)(m, k, n) }),
            F32 => Some(if n == Some(1) { (self.mmv_f32)(m, k) } else { (self.mmm_f32)(m, k, n) }),
            F16 => Some(if n == Some(1) { (self.mmv_f16)(m, k) } else { (self.mmm_f16)(m, k, n) }),
            I32 => {
                Some(if n == Some(1) { (self.qmmv_i32)(m, k) } else { (self.qmmm_i32)(m, k, n) })
            }
            _ => None,
        }
    }
}

pub fn generic() -> Ops {
    use crate::generic::mmm::*;
    Ops {
        mmm_impls: vec![generic_f32_4x4.mmm(), generic_f32_4x1.mmm()],
        mmm_f64: Box::new(|_, _, _| generic_f64_4x4.mmm()),
        mmv_f64: Box::new(|_, _| generic_f64_4x1.mmm()),
        mmm_f32: Box::new(|_, _, _| generic_f32_4x4.mmm()),
        mmv_f32: Box::new(|_, _| generic_f32_4x1.mmm()),
        mmm_f16: Box::new(|_, _, _| generic_f16_4x4.mmm()),
        mmv_f16: Box::new(|_, _| generic_f16_4x1.mmm()),
        qmmm_i32: Box::new(|_, _, _| generic_i32_4x4.mmm()),
        qmmv_i32: Box::new(|_, _| generic_i32_4x4.mmm()),
        leaky_relu_f16: Box::new(|| generic::HLeakyRelu8::ew()),
        leaky_relu_f32: Box::new(|| generic::SLeakyRelu4::ew()),
        mul_by_scalar_f16: Box::new(|| generic::HMulByScalar8::ew()),
        mul_by_scalar_f32: Box::new(|| generic::SMulByScalar4::ew()),
        sigmoid_f16: Box::new(|| generic::HSigmoid8::ew()),
        sigmoid_f32: Box::new(|| generic::SSigmoid4::ew()),
        tanh_f16: Box::new(|| generic::HTanh8::ew()),
        tanh_f32: Box::new(|| generic::STanh4::ew()),
        erf_f32: Box::new(|| generic::SErf4::ew()),
        lut_u8: Box::new(|table: &[u8]| Box::new(lut::LutImpl::<generic::GenericLut8>::new(table))),
        max_f16: Box::new(|| generic::reduce::max::HMax8::red()),
        max_f32: Box::new(|| generic::reduce::max::SMax4::red()),
        sum_f16: Box::new(|| generic::reduce::sum::HSum8::red()),
        sum_f32: Box::new(|| generic::reduce::sum::SSum4::red()),
        unicast_mul_f16: Box::new(|| generic::unicast::HUnicastMul8::bin()),
        unicast_mul_f32: Box::new(|| generic::unicast::SUnicastMul4::bin()),
        /*
        activation_f32: Box::new(|microcode| generic::SActivation::new(microcode))
        */
        softmax2_fastcompact_f16: Box::new(|| generic::reduce::softmax_l2::HSoftMaxL2::red()),
        softmax2_fastcompact_f32: Box::new(|| generic::reduce::softmax_l2::SSoftMaxL2::red()),
    }
}

#[allow(unreachable_code, unused_mut, unexpected_cfgs)]
pub fn best() -> Ops {
    let mut ops = generic();
    #[cfg(target_arch = "x86_64")]
    x86_64_fma::plug(&mut ops);
    #[cfg(any(target_arch = "arm", target_arch = "armv7"))]
    arm32::plug(&mut ops);
    #[cfg(target_arch = "aarch64")]
    arm64::plug(&mut ops);
    #[cfg(all(target_family = "wasm", target_feature = "simd128"))]
    wasm::plug(&mut ops);

    ops
}

lazy_static::lazy_static! {
    static ref OPS: Ops = {
        best()
    };
}

pub fn ops() -> &'static Ops {
    &OPS
}

use num_traits::*;
use std::fmt::Debug;
use std::ops::*;

pub trait LADatum:
    Sized
    + std::fmt::Display
    + Debug
    + Copy
    + Clone
    + Zero
    + One
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul
    + AddAssign
    + PartialOrd
    + Bounded
    + tract_data::prelude::Datum
{
    #[cfg(test)]
    fn strat() -> proptest::prelude::BoxedStrategy<Self>;
}

#[cfg(test)]
use proptest::prelude::*;

impl LADatum for f16 {
    #[cfg(test)]
    fn strat() -> BoxedStrategy<Self> {
        f32::strat().prop_map(|f| f.as_()).boxed()
    }
}

impl LADatum for f32 {
    #[cfg(test)]
    fn strat() -> BoxedStrategy<Self> {
        (-1000isize..1000).prop_map(|i| i as f32 / 1000.0).boxed()
    }
}

impl LADatum for f64 {
    #[cfg(test)]
    fn strat() -> BoxedStrategy<Self> {
        (-1000isize..1000).prop_map(|i| i as f64 / 1000.0).boxed()
    }
}

impl LADatum for u8 {
    #[cfg(test)]
    fn strat() -> BoxedStrategy<Self> {
        any::<u8>().boxed()
    }
}

impl LADatum for i8 {
    #[cfg(test)]
    fn strat() -> BoxedStrategy<Self> {
        any::<i8>().boxed()
    }
}

impl LADatum for i32 {
    #[cfg(test)]
    fn strat() -> BoxedStrategy<Self> {
        any::<i32>().boxed()
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn setup_test_logger() {
    let _ = env_logger::Builder::from_env("TRACT_LOG").try_init();
}

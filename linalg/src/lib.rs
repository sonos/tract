#[macro_use]
extern crate derive_new;
#[macro_use]
extern crate educe;
extern crate lazy_static;
extern crate libc;
extern crate log;
extern crate num_traits;
#[macro_use]
extern crate paste;
#[cfg(test)]
extern crate proptest;

include!(concat!(env!("OUT_DIR"), "/extern_kernel_macro.rs"));

#[macro_use]
pub mod frame;
mod generic;
pub use generic::ScaleShiftAndRound;
#[cfg(target_arch = "x86_64")]
pub mod x86_64_fma;

#[cfg(target_arch = "aarch64")]
pub mod arm64;

#[cfg(any(target_arch = "arm", target_arch = "armv7"))]
pub mod arm32;

pub use self::frame::{element_wise, lut, mmm};

use tract_data::prelude::*;

pub struct Ops {
    mmm_f32: Box<
        dyn Fn(Option<usize>, Option<usize>, Option<usize>) -> Box<dyn mmm::MatMatMul>
            + Send
            + Sync,
    >,
    mmv_f32: Box<dyn Fn(Option<usize>, Option<usize>) -> Box<dyn mmm::MatMatMul> + Send + Sync>,
    qmmm_i32: Box<
        dyn Fn(Option<usize>, Option<usize>, Option<usize>) -> Box<dyn mmm::MatMatMul>
            + Send
            + Sync,
    >,
    qmmv_i32: Box<dyn Fn(Option<usize>, Option<usize>) -> Box<dyn mmm::MatMatMul> + Send + Sync>,
    pub sigmoid_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub tanh_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub lut_u8: Box<dyn Fn(&[u8]) -> Box<dyn lut::Lut> + Send + Sync>,
    pub(crate) prefetch: Option<&'static (dyn Fn(*const u8, usize) + Sync + Send)>,
}

impl Ops {
    pub fn mmm(
        &self,
        a: DatumType,
        b: DatumType,
        c: DatumType,
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<Box<dyn mmm::MatMatMul>> {
        use DatumType::*;
        match (a.unquantized(), b.unquantized(), c.unquantized()) {
            (F32, F32, F32) => {
                Some(if n == Some(1) { (self.mmv_f32)(m, k) } else { (self.mmm_f32)(m, k, n) })
            }
            (I8, I8, I32) => {
                Some(if n == Some(1) { (self.qmmv_i32)(m, k) } else { (self.qmmm_i32)(m, k, n) })
            }
            (I8, I8, I8) => {
                Some(if n == Some(1) { (self.qmmv_i32)(m, k) } else { (self.qmmm_i32)(m, k, n) })
            }
            _ => None,
        }
    }
}

pub fn generic() -> Ops {
    Ops {
        mmm_f32: Box::new(|_, _, _| {
            Box::new(mmm::MatMatMulImpl::<generic::GenericMmm4x4<f32, f32, f32>, f32>::new())
        }),
        mmv_f32: Box::new(|_, _| {
            Box::new(mmm::MatMatMulImpl::<generic::GenericMmm4x1<f32, f32, f32>, f32>::new())
        }),
        qmmm_i32: Box::new(|_, _, _| {
            Box::new(mmm::MatMatMulImpl::<generic::GenericMmm4x4<i8, i8, i32>, i32>::new())
        }),
        qmmv_i32: Box::new(|_, _| {
            Box::new(mmm::MatMatMulImpl::<generic::GenericMmm4x1<i8, i8, i32>, i32>::new())
        }),
        sigmoid_f32: Box::new(|| {
            Box::new(element_wise::ElementWiseImpl::<generic::SSigmoid4, f32>::new())
        }),
        tanh_f32: Box::new(|| {
            Box::new(element_wise::ElementWiseImpl::<generic::STanh4, f32>::new())
        }),
        lut_u8: Box::new(|table: &[u8]| Box::new(lut::LutImpl::<generic::GenericLut8>::new(table))),
        prefetch: None,
    }
}

#[allow(unreachable_code, unused_mut)]
pub fn best() -> Ops {
    let mut ops = generic();
    #[cfg(target_arch = "x86_64")]
    x86_64_fma::plug(&mut ops);
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
mod test {
    use num_traits::*;
    use proptest::prelude::*;
    use std::fmt::Debug;
    use std::ops::*;

    pub trait LADatum:
        Sized
        + Debug
        + Copy
        + Clone
        + Zero
        + One
        + 'static
        + Add
        + Sub
        + Mul
        + AddAssign
        + MulAssign
        + PartialOrd
        + Bounded
        + tract_data::prelude::Datum
    {
        fn strat() -> BoxedStrategy<Self>;
        fn close(&self, other: &Self) -> bool;
    }

    impl LADatum for f32 {
        fn strat() -> BoxedStrategy<Self> {
            (-1000isize..1000).prop_map(|i| i as f32 / 1000.0).boxed()
        }
        fn close(&self, other: &Self) -> bool {
            (self - other).abs() < 0.001
        }
    }

    impl LADatum for u8 {
        fn strat() -> BoxedStrategy<Self> {
            any::<u8>().boxed()
        }
        fn close(&self, other: &Self) -> bool {
            self == other
        }
    }

    impl LADatum for i8 {
        fn strat() -> BoxedStrategy<Self> {
            any::<i8>().boxed()
        }
        fn close(&self, other: &Self) -> bool {
            self == other
        }
    }

    impl LADatum for i32 {
        fn strat() -> BoxedStrategy<Self> {
            any::<i32>().boxed()
        }
        fn close(&self, other: &Self) -> bool {
            self == other
        }
    }

    pub(crate) fn check_close<T: LADatum>(
        found: &[T],
        expected: &[T],
    ) -> proptest::test_runner::TestCaseResult {
        proptest::prop_assert!(
            found.iter().zip(expected.iter()).all(|(a, b)| a.close(b)),
            "found: {:?} expected: {:?}",
            found,
            expected
        );
        Ok(())
    }
}

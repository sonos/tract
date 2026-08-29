#![allow(clippy::missing_safety_doc)]
#![allow(clippy::redundant_closure_call)]
#![allow(clippy::len_zero)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(unexpected_cfgs)]
#![allow(unused_macros)]
#[macro_use]
extern crate derive_new;
extern crate lazy_static;
extern crate log;
extern crate num_traits;
#[macro_use]
extern crate pastey;
#[cfg(test)]
extern crate proptest;

include!(concat!(env!("OUT_DIR"), "/extern_kernel_macro.rs"));

/// Stands in for a function whose body only compiles in builds carrying the leading arch's
/// instructions — an asm block, an intrinsic, a CPUID probe — taking the argument types of
/// the real item and bailing when called. `wasm32` means wasm32 *with* `simd128`, the two
/// conditions the wasm kernels need. Needed only where something names the function on every
/// arch: a codegen macro, or a descriptor the arch tree declares everywhere while only the
/// native host ever calls what it names. A plain `#[cfg]` covers the rest.
macro_rules! bail_stub {
    (arm; $($rest:tt)*) => { bail_stub!(@ target_arch = "arm"; $($rest)*); };
    (aarch64; $($rest:tt)*) => { bail_stub!(@ target_arch = "aarch64"; $($rest)*); };
    (x86_64; $($rest:tt)*) => { bail_stub!(@ target_arch = "x86_64"; $($rest)*); };
    (riscv64; $($rest:tt)*) => { bail_stub!(@ target_arch = "riscv64"; $($rest)*); };
    (wasm32; $($rest:tt)*) => {
        bail_stub!(@ all(target_arch = "wasm32", target_feature = "simd128"); $($rest)*);
    };

    (@ $built:meta; $vis:vis unsafe fn $name:ident($($ty:ty),* $(,)?) $(-> $ret:ty)?) => {
        #[cfg(not($built))]
        $vis unsafe fn $name($(_: $ty),*) $(-> $ret)? {
            panic!(concat!(stringify!($name), ": not built for this target"))
        }
    };

    (@ $built:meta; $vis:vis fn $name:ident($($ty:ty),* $(,)?) $(-> $ret:ty)?) => {
        #[cfg(not($built))]
        $vis fn $name($(_: $ty),*) $(-> $ret)? {
            panic!(concat!(stringify!($name), ": not built for this target"))
        }
    };
}

#[macro_use]
mod frame;
#[macro_use]
pub mod routines;
pub mod cache;
pub mod generic;
pub mod knobs;
pub mod multithread;
pub use frame::weights::WeightType;
pub use generic::{ScaleShiftAndRound, Scaler};
use tract_data::internal::TensorView;
// An arch tree compiles when this build can run its kernels, and — for enumeration only —
// when `foreign-inventory` asks for the others as well.
#[cfg(any(target_arch = "x86_64", feature = "foreign-inventory"))]
pub mod x86_64;

pub mod hwbench;

#[cfg(any(target_arch = "aarch64", feature = "foreign-inventory"))]
pub mod arm64;

#[cfg(any(target_arch = "aarch64", feature = "foreign-inventory"))]
pub use arm64::has_fp16;

/// True when the running CPU implements FEAT_FP16. No arm64 tree in this build, hence no
/// kernel that could use it.
#[cfg(not(any(target_arch = "aarch64", feature = "foreign-inventory")))]
pub fn has_fp16() -> bool {
    false
}

#[cfg(any(target_arch = "arm", feature = "foreign-inventory"))]
pub mod arm32;

#[cfg(any(all(target_arch = "wasm32", target_feature = "simd128"), feature = "foreign-inventory"))]
pub mod wasm;

pub mod isa;
pub mod mmm_routines;
pub mod mmm_tiers;

pub use self::frame::mmm::MmmDispatch;
pub use self::frame::*;
pub use self::routines::Func;

use tract_data::prelude::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinOp {
    Min,
    Max,
    Add,
    Mul,
    Sub,
    SubF,
}

impl BinOp {
    pub fn flip(&self) -> BinOp {
        use BinOp::*;
        match self {
            Sub => SubF,
            SubF => Sub,
            sym => *sym,
        }
    }
}

/// A binary operation over two tensor views, writing its result over the left one. What the two
/// binary layouts erase to -- [`by_scalar::ByScalarKer::bin`] broadcasts a one-element right
/// operand, [`unicast::UnicastKer::bin`] walks a right operand of the same length -- so a caller
/// holding one needs to know neither which layout nor which kernel answered.
pub type BinFn = dyn Fn(&mut TensorView, &TensorView) -> TractResult<()> + Send + Sync;
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


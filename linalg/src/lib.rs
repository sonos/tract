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
/// arch: a codegen macro, or a `plug` the arch tree compiles everywhere but nothing except
/// the native host ever calls. A plain `#[cfg]` covers the rest.
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
pub mod cache;
pub mod generic;
pub mod knobs;
pub mod multithread;
pub use frame::weights::WeightType;
pub use generic::{ScaleShiftAndRound, Scaler};
use lazy_static::lazy_static;
use mmm::{
    ImplementationQuality, MMMInputFormat, Query, Suitable, pick_by_shape, retain_best_quality,
};
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

use tract_itertools::Itertools;

#[cfg(any(target_arch = "arm", feature = "foreign-inventory"))]
pub mod arm32;

#[cfg(any(all(target_arch = "wasm32", target_feature = "simd128"), feature = "foreign-inventory"))]
pub mod wasm;

pub mod isa;
pub mod mmm_routines;
pub mod platform;

pub use self::frame::*;

use tract_data::prelude::*;

/// One tier’s answer within a policy: which of the suitable kernels it would run, `None` to leave the
/// query to the tier below.
pub type MmmPreference = Box<dyn Fn(&[Suitable], &Query) -> Option<usize> + Send + Sync>;

/// Which of the suitable kernels a platform would run for a query, `None` for a query it has no
/// opinion on — an accumulator no arch plug claimed, or a shape whose kernel this build does not
/// offer. The answer is an index, so a policy can only ever name one the query reached.
pub type MmmPolicy = Box<dyn Fn(DatumType, &Query, &[Suitable]) -> Option<usize> + Send + Sync>;

#[allow(clippy::type_complexity)]
pub struct Ops {
    runnable: Vec<Box<dyn mmm::MatMatMul>>,
    panel_extractors: Vec<mmm::PanelExtractor>,

    mmm_policy: MmmPolicy,

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
    pub hardswish_f16: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16>> + Send + Sync>,
    pub hardswish_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub silu_f16: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16>> + Send + Sync>,
    pub silu_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub gelu_f16: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f16>> + Send + Sync>,
    pub gelu_f32: Box<dyn Fn() -> Box<dyn element_wise::ElementWise<f32>> + Send + Sync>,
    pub lut_u8: Box<dyn Fn(&[u8]) -> Box<dyn lut::Lut> + Send + Sync>,

    pub max_f16: Box<dyn Fn() -> Box<dyn reduce::Reduce<f16>> + Send + Sync>,
    pub max_f32: Box<dyn Fn() -> Box<dyn reduce::Reduce<f32>> + Send + Sync>,
    pub min_f32: Box<dyn Fn() -> Box<dyn reduce::Reduce<f32>> + Send + Sync>,

    pub sum_f16: Box<dyn Fn() -> Box<dyn reduce::Reduce<f16>> + Send + Sync>,
    pub sum_f32: Box<dyn Fn() -> Box<dyn reduce::Reduce<f32>> + Send + Sync>,

    pub softmax2_f32: Box<dyn Fn() -> Box<dyn reduce::MapReduce<f32, f32>> + Send + Sync>,

    /// Fused row-wise RmsNorm: out_i = x_i * rsqrt(mean(x_i²) + eps).
    /// Replaces a 4-call composition (MeanOfSquares + Add + Rsqrt + Mul) with
    /// a single 2-pass kernel. Called once per row by `core::ops::nn::RmsNorm`
    /// when the input is f32 and the axis is the last (contiguous) one.
    pub rms_norm_f32: Box<dyn Fn(&mut [f32], f32) + Send + Sync>,
}

impl Ops {
    /// Every kernel this host can execute: built into this build, and declaring an instruction
    /// set the CPU has. What selection narrows down from.
    pub fn runnable(&self) -> &[Box<dyn mmm::MatMatMul>] {
        &self.runnable
    }

    pub fn all_possible_packing(
        &self,
        weight_type: impl Into<WeightType>,
    ) -> impl Iterator<Item = &dyn MMMInputFormat> {
        let weight_type = weight_type.into();
        self.runnable
            .iter()
            .flat_map(|m| m.packings())
            .map(|p| &*p.0)
            .flat_map(move |p| {
                let mut packs: Vec<&dyn MMMInputFormat> = vec![];
                if p.precursor() == weight_type {
                    packs.push(p)
                };
                for pe in &self.panel_extractors {
                    if pe.from.precursor() == weight_type && pe.to.dyn_eq(p) {
                        packs.push(&*pe.from);
                    }
                }
                packs.into_iter()
            })
            .sorted_by_key(|p| p.to_string())
            .dedup()
    }

    /// Every way this build can compute the queried matmul: a kernel, which of its packings
    /// to use, and the panel extractor to reach that packing when the weights are not
    /// already in it. The query’s dims are not consulted — a kernel is suitable or not
    /// whatever the shape.
    ///
    /// The one enumeration behind both matmul lowerings — how one is *chosen* from
    /// the list is the caller's business.
    pub fn suitable(&self, query: &Query) -> Vec<Suitable> {
        self.runnable
            .iter()
            .filter(|mmm| {
                query.accumulators.contains(&mmm.internal_type())
                    && query.store.is_none_or(|s| mmm.stores().contains(&s))
            })
            .flat_map(|mmm| mmm.packings().iter().enumerate().map(move |(ix, p)| (mmm, ix, p)))
            .filter(|(_, _, (_, b))| {
                b.precursor().as_dt().is_some_and(|dt| dt == query.activation.unquantized())
            })
            .filter_map(|(mmm, ix, (a, _))| {
                if a.precursor() == query.weight {
                    Some((mmm.clone(), ix, None))
                } else if query.allow_extractor {
                    self.panel_extractors
                        .iter()
                        .find(|pe| pe.from.precursor() == query.weight && pe.to.dyn_eq(&**a))
                        .map(|pe| (mmm.clone(), ix, Some(pe.clone())))
                } else {
                    None
                }
            })
            .collect()
    }

    /// The platform policy's choice among `suitable`, or `None` when it has no opinion —
    /// no arch plug claimed this accumulator, or the query is not the plain matmul the policies
    /// reason about. A generic answer counts as no opinion: `generic()` installs fixed kernels,
    /// so getting one back means no arch plug ever overwrote that slot.
    pub fn preferred(&self, query: &Query, suitable: &[Suitable]) -> Option<usize> {
        let WeightType::Plain(weight) = &query.weight else { return None };
        if weight.unquantized() != query.activation.unquantized() {
            return None;
        }
        let ix = (self.mmm_policy)(*query.accumulators.first()?, query, suitable)?;
        let quality = suitable[ix].0.quality();
        (quality == ImplementationQuality::ManuallyOptimized).then_some(ix)
    }

    /// One kernel for the query, for a caller that needs an answer now: the platform policy's
    /// pick where it has an opinion, then the portable rules, then the widest extractor-free
    /// tile. That last resort is what a caller with no fallback of its own needs when `n` is
    /// unknown or degenerate — a caller that can do better with the whole list, as einsum can
    /// for a symbolic `n`, should walk the suitable kernels itself. `None` only when nothing suitable
    /// exists at all.
    pub fn pick(&self, query: &Query) -> Option<Suitable> {
        let mut suitable = self.suitable(query);
        if let Some(ix) = self.preferred(query, &suitable) {
            return Some(suitable.swap_remove(ix));
        }
        retain_best_quality(&mut suitable);
        if suitable.len() == 1 {
            return Some(suitable.remove(0));
        }
        if let Some(ix) = pick_by_shape(query, &suitable) {
            return Some(suitable.swap_remove(ix));
        }
        let ix = suitable
            .iter()
            .enumerate()
            .max_by_key(|(_, (mmm, _, pe))| (pe.is_none(), mmm.nr() > 1, mmm.nr() * mmm.mr()))
            .map(|(ix, _)| ix)?;
        Some(suitable.swap_remove(ix))
    }

    pub fn panel_extractors(&self) -> &[mmm::panel_extract::PanelExtractor] {
        &self.panel_extractors
    }

    /// The kernel this platform's policy would run for a plain matmul of these dims, for a
    /// caller introspecting dispatch rather than performing it. Unlike [`Ops::preferred`] it reports
    /// the policy's answer whatever its quality tier, and it never falls back on the portable
    /// rules: `None` means the policy itself had nothing to say.
    pub fn preferred_kernel(
        &self,
        accumulator: DatumType,
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<Box<dyn mmm::MatMatMul>> {
        let query = Query::plain(accumulator, m, k, n);
        let suitable = self.suitable(&query);
        let ix = (self.mmm_policy)(accumulator, &query, &suitable)?;
        Some(suitable[ix].0.clone())
    }

    /// Put `f` in front of the policy: it answers the accumulators and shapes it claims, and
    /// defers everything else to what was there before. This is how an arch tier layers over
    /// the tiers below it, and how a tier that speaks for one accumulator leaves the others be.
    pub fn overlay_mmm_policy(
        &mut self,
        f: impl Fn(&MmmPolicy, DatumType, &Query, &[Suitable]) -> Option<usize> + Send + Sync + 'static,
    ) {
        let prev = std::mem::replace(&mut self.mmm_policy, Box::new(|_, _, _| None));
        self.mmm_policy = Box::new(move |dt, query, suitable| f(&prev, dt, query, suitable));
    }
}

pub fn generic() -> Ops {
    use crate::generic::mmm::*;
    use crate::mmm::suitable_named;
    use element_wise::ElementWiseKer;
    use reduce::{MapReduceKer, ReduceKer};
    let mut ops = Ops {
        runnable: vec![],
        panel_extractors: vec![],
        mmm_policy: Box::new(|dt, query, suitable| {
            let vec = query.n == Some(1);
            let name = match dt {
                DatumType::F64 if vec => &generic_f64_4x1.name,
                DatumType::F64 => &generic_f64_4x4.name,
                DatumType::F32 if vec => &generic_f32_4x1.name,
                DatumType::F32 => &generic_f32_4x4.name,
                DatumType::F16 if vec => &generic_f16_4x1.name,
                DatumType::F16 => &generic_f16_4x4.name,
                DatumType::I32 => &generic_i32_4x4.name,
                _ => return None,
            };
            suitable_named(suitable, name)
        }),
        leaky_relu_f16: Box::new(|| generic::HLeakyRelu8::ew()),
        leaky_relu_f32: Box::new(|| generic::SLeakyRelu4::ew()),
        mul_by_scalar_f16: Box::new(|| generic::HMulByScalar8::ew()),
        mul_by_scalar_f32: Box::new(|| generic::SMulByScalar4::ew()),
        sigmoid_f16: Box::new(|| generic::HSigmoid8::ew()),
        sigmoid_f32: Box::new(|| generic::SSigmoid4::ew()),
        tanh_f16: Box::new(|| generic::HTanh8::ew()),
        tanh_f32: Box::new(|| generic::STanh4::ew()),
        erf_f32: Box::new(|| generic::SErf4::ew()),
        hardswish_f16: Box::new(|| generic::HHardSwish8::ew()),
        hardswish_f32: Box::new(|| generic::SHardSwish4::ew()),
        silu_f16: Box::new(|| generic::HSiLU8::ew()),
        silu_f32: Box::new(|| generic::SSiLU4::ew()),
        gelu_f16: Box::new(|| generic::HGelu8::ew()),
        gelu_f32: Box::new(|| generic::SGelu4::ew()),
        lut_u8: Box::new(|table: &[u8]| Box::new(lut::LutImpl::<generic::GenericLut8>::new(table))),
        max_f16: Box::new(|| generic::reduce::max::HMax8::red()),
        max_f32: Box::new(|| generic::reduce::max::SMax4::red()),
        min_f32: Box::new(|| generic::reduce::min::SMin4::red()),
        sum_f16: Box::new(|| generic::reduce::sum::HSum8::red()),
        sum_f32: Box::new(|| generic::reduce::sum::SSum4::red()),
        /*
        activation_f32: Box::new(|microcode| generic::SActivation::new(microcode))
        */
        softmax2_f32: Box::new(|| generic::reduce::softmax_l2::SSoftMaxL2Accurate::red()),
        rms_norm_f32: Box::new(generic::rms_norm::rms_norm_f32),
    };
    ops.runnable = mmm_routines::runnable();
    ops
}

pub fn best() -> Ops {
    let mut ops = generic();
    for selector in platform::all().filter(|s| s.target.is_native()) {
        (selector.plug)(&mut ops);
    }
    ops
}

lazy_static::lazy_static! {
    static ref OPS: Ops = {
        best()
    };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

fn register_all_unicast(registry: &mut LinalgRegistry) {
    generic::register_all_unicast(registry);
    #[cfg(target_arch = "aarch64")]
    arm64::register_all_unicast(registry);
}

fn register_all_by_scalar(registry: &mut LinalgRegistry) {
    generic::register_all_by_scalar(registry);
    #[cfg(target_arch = "aarch64")]
    arm64::register_all_by_scalar(registry);
}

pub type LinalgFn = dyn Fn(&mut TensorView, &TensorView) -> TractResult<()> + Send + Sync;
type LinalgRegistry = HashMap<(BinOp, DatumType), Box<dyn Fn() -> Box<LinalgFn> + Send + Sync>>;
lazy_static! {
    static ref BIN_UNICAST_OPS: Mutex<LinalgRegistry> = {
        let mut registry = HashMap::default();
        register_all_unicast(&mut registry);
        Mutex::new(registry)
    };
    static ref BIN_BY_SCALAR_OPS: Mutex<LinalgRegistry> = {
        let mut registry = HashMap::default();
        register_all_by_scalar(&mut registry);
        Mutex::new(registry)
    };
}

pub fn bin_by_scalar(dt: DatumType, bin: BinOp) -> Option<Box<LinalgFn>> {
    let map = BIN_BY_SCALAR_OPS.lock().unwrap();
    if (dt == DatumType::F16) && !has_fp16() {
        return None;
    }
    map.get(&(bin, dt)).map(|it| (it)())
}

pub fn bin_unicast(dt: DatumType, bin: BinOp) -> Option<Box<LinalgFn>> {
    let map = BIN_UNICAST_OPS.lock().unwrap();
    if (dt == DatumType::F16) && !has_fp16() {
        return None;
    }
    map.get(&(bin, dt)).map(|it| (it)())
}

pub fn ops() -> &'static Ops {
    &OPS
}

use dyn_eq::DynEq;
use num_traits::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::*;
use std::sync::Mutex;

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

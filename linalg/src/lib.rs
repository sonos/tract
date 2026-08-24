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
#[macro_use]
pub mod routines;
pub mod cache;
pub mod generic;
pub mod knobs;
pub mod multithread;
pub use frame::weights::WeightType;
pub use generic::{ScaleShiftAndRound, Scaler};
use isa::{Arch, IsaSet};
use lazy_static::lazy_static;
use mmm::{
    ImplementationQuality, MMMInputFormat, Query, Suitable, pick_by_shape, retain_best_quality,
};
use mmm_tiers::MmmTier;
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
pub mod mmm_tiers;

pub use self::frame::*;

use tract_data::prelude::*;

#[allow(clippy::type_complexity)]
pub struct Ops {
    /// The architecture these kernels are for and what its instruction set offers. Everything mmm
    /// selection is a function of it: the runnable set, and which tiers speak.
    isa: IsaSet,
    /// The applicable tiers, highest precedence first, resolved once from [`Self::isa`].
    tiers: Vec<&'static MmmTier>,
    runnable: Vec<Box<dyn mmm::MatMatMul>>,
    panel_extractors: Vec<mmm::PanelExtractor>,

    pub lut_u8: Box<dyn Fn(&[u8]) -> Box<dyn lut::Lut> + Send + Sync>,
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

    /// This platform’s tiers, highest precedence first: the ladder [`Self::preferred`] walks.
    pub fn tiers(&self) -> &[&'static MmmTier] {
        &self.tiers
    }

    /// The platform's choice among `suitable`, or `None` when no tier has an opinion — none
    /// claimed this accumulator, or the query is not the plain matmul the tiers reason about.
    /// A generic answer counts as no opinion: the precedence-0 tier answers for every platform, so
    /// getting one of its kernels back means no arch tier claimed the query.
    pub fn preferred(&self, query: &Query, suitable: &[Suitable]) -> Option<usize> {
        let WeightType::Plain(weight) = &query.weight else { return None };
        if weight.unquantized() != query.activation.unquantized() {
            return None;
        }
        let acc = *query.accumulators.first()?;
        let ix = mmm_tiers::preferred(&self.isa, &self.tiers, acc, query, suitable)?;
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

    /// The kernel this platform would run for a plain matmul of these dims, for a caller
    /// introspecting dispatch rather than performing it. Unlike [`Ops::preferred`] it reports
    /// the tiers' answer whatever its quality, and it never falls back on the portable rules:
    /// `None` means no tier had anything to say.
    pub fn preferred_kernel(
        &self,
        accumulator: DatumType,
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<Box<dyn mmm::MatMatMul>> {
        let query = Query::plain(accumulator, m, k, n);
        let suitable = self.suitable(&query);
        let ix = mmm_tiers::preferred(&self.isa, &self.tiers, accumulator, &query, &suitable)?;
        Some(suitable[ix].0.clone())
    }
}

/// The portable rules, the tier every platform ends on: fixed generic kernels, in the tile the
/// shape asks for. Rank 0, so any arch tier that claims the query answers before it.
fn generic_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    use crate::generic::mmm::*;
    let vec = query.n == Some(1);
    let name = match dt {
        DatumType::F64 if vec => generic_f64_4x1.name.as_str(),
        DatumType::F64 => generic_f64_4x4.name.as_str(),
        DatumType::F32 if vec => generic_f32_4x1.name.as_str(),
        DatumType::F32 => generic_f32_4x4.name.as_str(),
        DatumType::F16 if vec => generic_f16_4x1.name.as_str(),
        DatumType::F16 => generic_f16_4x4.name.as_str(),
        DatumType::I32 => generic_i32_4x4.name.as_str(),
        _ => return None,
    };
    Some(name)
}

inventory::submit! {
    MmmTier {
        arch: None,
        precedence: 0,
        name: "generic",
        applies: |_| true,
        preferred: generic_preferred,
    }
}

/// The portable `Ops`, on the platform it is asked for: its runnable kernels and the tiers that
/// speak for it. Both are functions of the platform, so mmm dispatch is settled here; what an
/// arch `plug` still adds is the non-mmm kernel slots.
pub fn generic_for(isa: IsaSet) -> Ops {
    let mut ops = Ops {
        isa,
        tiers: mmm_tiers::for_isa(&isa),
        runnable: vec![],
        panel_extractors: vec![],
        lut_u8: Box::new(|table: &[u8]| Box::new(lut::LutImpl::<generic::GenericLut8>::new(table))),
        /*
        activation_f32: Box::new(|microcode| generic::SActivation::new(microcode))
        */
    };
    ops.runnable = match isa.arch() {
        Some(target) => mmm_routines::runnable_for(target),
        None => mmm_routines::runnable(),
    };
    ops
}

/// The portable `Ops` for this host.
pub fn generic() -> Ops {
    generic_for(isa::native())
}

/// What an arch tree still installs by hand: the non-mmm kernel slots. mmm dispatch needs none of
/// this — it is a function of the instruction set, through [`mmm_tiers`] and [`mmm_routines`].
pub struct ArchPlug {
    /// Arch the slots are written for. Not an `Option`, unlike
    /// [`mmm_routines::MmmRoutine::target`]: a kernel can be portable, a tree is always
    /// somebody's.
    pub arch: Arch,
    pub plug: fn(&mut Ops),
}

inventory::collect!(ArchPlug);

/// Every arch tree compiled into this build, native or not.
pub fn arch_plugs() -> impl Iterator<Item = &'static ArchPlug> {
    inventory::iter::<ArchPlug>()
}

pub fn best() -> Ops {
    let mut ops = generic();
    for plug in arch_plugs().filter(|p| p.arch.is_native()) {
        (plug.plug)(&mut ops);
    }
    ops
}

/// `Ops` as `arch` sees them: its kernels, from [`mmm_routines::runnable_for`], under its own
/// tiers. Answers which kernel that architecture would choose for a shape, from any host. What it
/// cannot reproduce is a hardware probe, so for a foreign arch it starts from the plain
/// architecture and nothing else: a cohort behind a feature is reached by naming that feature in
/// `TRACT_CPU_ISA`, which is checked against this architecture rather than the host's. `None` when
/// the architecture's tree was not compiled in; see the `foreign-inventory` feature.
pub fn inspect(arch: Arch) -> Option<Ops> {
    if !mmm_routines::declared().any(|r| r.target == Some(arch)) {
        return None;
    }
    let isa = if arch.is_native() { isa::native() } else { isa::forced(IsaSet::of_arch(arch)) };
    let mut ops = generic_for(isa);
    for plug in arch_plugs().filter(|p| p.arch == arch) {
        (plug.plug)(&mut ops);
    }
    Some(ops)
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

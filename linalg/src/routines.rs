//! PROTOTYPE — single-source-of-truth registry for linalg routines: pointwise
//! activations, reductions (and, in progress, more kernel families).
//!
//! Each concrete kernel submits one [`RoutineImpl`] descriptor through
//! [`inventory`], from a per-backend `routines.rs` file that sits next to the
//! kernels it describes. The descriptor is pure data (function, dtype, target,
//! tier, kernel name, feature probe) plus an optional `factory`. The factory is
//! `Some` only in a build that targets the kernel's own architecture; a build that
//! merely *describes* a foreign architecture (see the `registry-all-targets`
//! feature) carries the descriptor with `factory: None`, so no foreign symbol is
//! ever named.
//!
//! This yields three states per cell of the (function × target) matrix:
//! * **bound**     — descriptor present, `factory` set, feature probe passes on this host
//! * **described** — descriptor present, but not runnable here (foreign arch, or the
//!   feature is absent on this host)
//! * **absent**    — the descriptor was not compiled into this build at all
//!
//! Selection ([`pick`]) is order-independent: among the *bound* descriptors for a
//! `(func, dtype)` it takes the one with the highest `(tier, isa_rank)`. This
//! replaces the last-writer-wins field assignments in each arch's `plug()`.

use crate::frame::element_wise::ElementWise;
use crate::frame::reduce::{MapReduce, Reduce};
use crate::{BinOp, LinalgFn};
use tract_data::prelude::{DatumType, f16};

/// Which activation function a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Routine {
    Sigmoid,
    Silu,
    Tanh,
    Erf,
    HardSwish,
    Gelu,
    /// Takes a runtime `alpha` param (`ElementWise<T, T>`); dispatched via
    /// [`kernel_f32_param`] / [`kernel_f16_param`].
    LeakyRelu,
    /// Multiply by a runtime scalar (`ElementWise<T, T>`); param-dispatched like
    /// [`Routine::LeakyRelu`].
    MulByScalar,
    // Reductions. Dispatched via `reduce_*` / `map_reduce_*`. Prefixed to stay
    // distinct from the binary `BinOp::{Max,Min}`.
    ReduceMax,
    ReduceMin,
    ReduceSum,
    Softmax,
    /// Binary op, tensor ⊙ scalar (broadcast). Dispatched via [`bin`].
    BinByScalar(BinOp),
    /// Binary op, elementwise tensor ⊙ tensor. Dispatched via [`bin`].
    BinUnicast(BinOp),
}

/// How directly a kernel implements its function, best first. The variant order is
/// the primary selection key: a `Native` kernel always beats any `Via(_)` composition,
/// which beats the portable `Generic` reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Portable reference implementation; available on every target.
    Generic,
    /// Reachable but indirect, routed through the named kernel — `Via("f32")` for an
    /// f16 kernel that round-trips through an f32 one, `Via("sigmoid")` for a silu
    /// built from sigmoid. A dedicated kernel may beat it.
    Via(&'static str),
    /// Hand-written kernel for exactly this (function, dtype, target).
    Native,
}

/// A boxed factory for the concrete kernel, monomorphized per element type.
/// Present only when the descriptor's target matches the compile target. The
/// `*Param` variants are for activations that take a runtime scalar param (e.g.
/// leaky-relu's `alpha`), whose `ElementWise<T, T>` is a distinct type.
#[derive(Clone, Copy)]
pub enum RoutineFactory {
    F32(fn() -> Box<dyn ElementWise<f32>>),
    F16(fn() -> Box<dyn ElementWise<f16>>),
    F32Param(fn() -> Box<dyn ElementWise<f32, f32>>),
    F16Param(fn() -> Box<dyn ElementWise<f16, f16>>),
    F32Reduce(fn() -> Box<dyn Reduce<f32>>),
    F16Reduce(fn() -> Box<dyn Reduce<f16>>),
    F32MapReduce(fn() -> Box<dyn MapReduce<f32, f32>>),
    F16MapReduce(fn() -> Box<dyn MapReduce<f16, f16>>),
    /// Binary op over two tensor views (by-scalar or unicast); dtype-erased.
    Bin(fn() -> Box<LinalgFn>),
}

/// One registered activation kernel. Submitted via [`inventory`] next to its kernel.
pub struct RoutineImpl {
    pub func: Routine,
    pub dt: DatumType,
    /// Coarse target axis for the matrix, e.g. `"generic"`, `"aarch64"`, `"x86_64"`.
    pub target: &'static str,
    /// CPU feature the kernel needs within `target`, for display (`None` = baseline).
    pub feature: Option<&'static str>,
    pub tier: Tier,
    /// Intra-tier preference (higher ISA wins), e.g. avx < fma < avx512f.
    pub isa_rank: u8,
    /// Kernel symbol name; must equal the `ElementWise::name()` it produces.
    pub kernel: &'static str,
    /// Whether this kernel is runnable on the host executing *now*. `false` for any
    /// descriptor whose target is not the running arch.
    pub check: fn() -> bool,
    pub factory: Option<RoutineFactory>,
}

impl RoutineImpl {
    /// Runnable on this host right now: compiled in *and* its feature probe passes.
    pub fn is_bound(&self) -> bool {
        self.factory.is_some() && (self.check)()
    }
}

inventory::collect!(RoutineImpl);

/// Every registered descriptor (bound, described, across all compiled-in targets).
pub fn all() -> impl Iterator<Item = &'static RoutineImpl> {
    inventory::iter::<RoutineImpl>()
}

/// The kernel the engine would dispatch to for `(func, dt)` on this host: the
/// highest-`(tier, isa_rank)` descriptor that is bound. `None` if nothing is bound
/// (should never happen while the generic tier is compiled in).
pub fn pick(func: Routine, dt: DatumType) -> Option<&'static RoutineImpl> {
    all()
        .filter(|a| a.func == func && a.dt == dt && a.is_bound())
        .max_by_key(|a| (a.tier, a.isa_rank))
}

/// The `f32` kernel to run for `func` on this host. The generic tier guarantees a
/// bound kernel, so this is infallible for every registered `f32` activation.
pub fn kernel_f32(func: Routine) -> Box<dyn ElementWise<f32>> {
    match pick(func, DatumType::F32).and_then(|a| a.factory) {
        Some(RoutineFactory::F32(make)) => make(),
        _ => unreachable!("no bound f32 kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f16` kernel to run for `func` on this host. Infallible for every `func`
/// that has an `f16` form registered (all but [`Routine::Erf`]).
pub fn kernel_f16(func: Routine) -> Box<dyn ElementWise<f16>> {
    match pick(func, DatumType::F16).and_then(|a| a.factory) {
        Some(RoutineFactory::F16(make)) => make(),
        _ => unreachable!("no bound f16 kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f32` kernel for a parameterized activation (e.g. [`Routine::LeakyRelu`]),
/// run with `run_with_params(xs, alpha)`.
pub fn kernel_f32_param(func: Routine) -> Box<dyn ElementWise<f32, f32>> {
    match pick(func, DatumType::F32).and_then(|a| a.factory) {
        Some(RoutineFactory::F32Param(make)) => make(),
        _ => unreachable!("no bound f32 param kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f16` kernel for a parameterized activation, run with
/// `run_with_params(xs, alpha)`.
pub fn kernel_f16_param(func: Routine) -> Box<dyn ElementWise<f16, f16>> {
    match pick(func, DatumType::F16).and_then(|a| a.factory) {
        Some(RoutineFactory::F16Param(make)) => make(),
        _ => unreachable!("no bound f16 param kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f32` reducer for `func` (e.g. [`Routine::ReduceMax`]).
pub fn reduce_f32(func: Routine) -> Box<dyn Reduce<f32>> {
    match pick(func, DatumType::F32).and_then(|a| a.factory) {
        Some(RoutineFactory::F32Reduce(make)) => make(),
        _ => unreachable!("no bound f32 reducer for {func:?} (generic tier missing?)"),
    }
}

/// The `f16` reducer for `func`.
pub fn reduce_f16(func: Routine) -> Box<dyn Reduce<f16>> {
    match pick(func, DatumType::F16).and_then(|a| a.factory) {
        Some(RoutineFactory::F16Reduce(make)) => make(),
        _ => unreachable!("no bound f16 reducer for {func:?} (generic tier missing?)"),
    }
}

/// The `f32` map-reducer for `func` (e.g. [`Routine::Softmax`]).
pub fn map_reduce_f32(func: Routine) -> Box<dyn MapReduce<f32, f32>> {
    match pick(func, DatumType::F32).and_then(|a| a.factory) {
        Some(RoutineFactory::F32MapReduce(make)) => make(),
        _ => unreachable!("no bound f32 map-reducer for {func:?} (generic tier missing?)"),
    }
}

/// The `f16` map-reducer for `func`.
pub fn map_reduce_f16(func: Routine) -> Box<dyn MapReduce<f16, f16>> {
    match pick(func, DatumType::F16).and_then(|a| a.factory) {
        Some(RoutineFactory::F16MapReduce(make)) => make(),
        _ => unreachable!("no bound f16 map-reducer for {func:?} (generic tier missing?)"),
    }
}

/// The binary-op kernel for `func` (a [`Routine::BinByScalar`] / [`Routine::BinUnicast`])
/// at `dt`, or `None` if none is bound on this host — e.g. f16 without hardware fp16,
/// which mirrors the old `bin_by_scalar`/`bin_unicast` fallback to the scalar path.
pub fn bin(func: Routine, dt: DatumType) -> Option<Box<LinalgFn>> {
    match pick(func, dt).and_then(|a| a.factory) {
        Some(RoutineFactory::Bin(make)) => Some(make()),
        _ => None,
    }
}

// Descriptors are `pub mod routines` submodules of each backend module (generic,
// arm64, x86_64_fma, arm32, wasm) — co-located with the kernels they describe. The
// arch module compiles for its own target OR under `registry-all-targets`, so those
// descriptors are visible to a host drawing the whole matrix even though the arch's
// native kernels are `#[cfg(target_arch)]`-gated out.

#[cfg(test)]
mod test {
    use super::*;

    /// Every registered activation resolves to a bound kernel on this host — the
    /// generic tier is the floor, so dispatch can never come up empty.
    #[test]
    fn every_activation_dispatches() {
        crate::setup_test_logger();
        let f32s = [
            Routine::Sigmoid,
            Routine::Silu,
            Routine::Tanh,
            Routine::Erf,
            Routine::HardSwish,
            Routine::Gelu,
        ];
        for func in f32s {
            assert!(pick(func, DatumType::F32).is_some(), "no bound f32 kernel for {func:?}");
        }
        // Erf has no f16 form.
        for func in [
            Routine::Sigmoid,
            Routine::Silu,
            Routine::Tanh,
            Routine::HardSwish,
            Routine::Gelu,
            Routine::LeakyRelu,
            Routine::MulByScalar,
        ] {
            assert!(pick(func, DatumType::F16).is_some(), "no bound f16 kernel for {func:?}");
        }
        for func in [
            Routine::LeakyRelu,
            Routine::MulByScalar,
            Routine::ReduceMax,
            Routine::ReduceMin,
            Routine::ReduceSum,
            Routine::Softmax,
        ] {
            assert!(pick(func, DatumType::F32).is_some(), "no bound f32 kernel for {func:?}");
        }
        for func in [Routine::ReduceMax, Routine::ReduceSum, Routine::Softmax] {
            assert!(pick(func, DatumType::F16).is_some(), "no bound f16 kernel for {func:?}");
        }
    }

    /// The picked reducers compute the right scalar (max and sum over a slice).
    #[test]
    fn reducers_run() {
        let xs = [1.0f32, -3.0, 2.5, 0.0, 4.0];
        let max = reduce_f32(Routine::ReduceMax).run(&xs).unwrap();
        assert!((max - 4.0).abs() < 1e-5, "max={max}");
        let sum = reduce_f32(Routine::ReduceSum).run(&xs).unwrap();
        assert!((sum - 4.5).abs() < 1e-4, "sum={sum}");
    }

    /// The picked leaky-relu kernel applies `alpha` on the negative side.
    #[test]
    fn leaky_kernel_runs() {
        let op = kernel_f32_param(Routine::LeakyRelu);
        let mut xs = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        op.run_with_params(&mut xs, 0.1).unwrap();
        for (x, y) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().zip(&xs) {
            let want = if *x >= 0.0 { *x } else { *x * 0.1 };
            assert!((want - y).abs() < 1e-4, "leaky_relu({x})={y} want {want}");
        }
    }

    /// The picked kernel actually computes sigmoid.
    #[test]
    fn picked_kernel_runs() {
        let d = pick(Routine::Sigmoid, DatumType::F32).unwrap();
        let Some(RoutineFactory::F32(f)) = d.factory else { panic!("f32 factory") };
        let mut xs = vec![0f32, 1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 2.0];
        f().run(&mut xs).unwrap();
        for (x, y) in [0f32, 1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 2.0].iter().zip(&xs) {
            let want = 1.0 / (1.0 + (-x).exp());
            assert!((want - y).abs() < 1e-3, "sigmoid({x})={y} want {want}");
        }
    }
}

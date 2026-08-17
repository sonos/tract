//! PROTOTYPE — single-source-of-truth registry for element-wise activation kernels.
//!
//! Each concrete kernel submits one [`ActivationImpl`] descriptor through
//! [`inventory`], from a per-backend `activation.rs` file that sits next to the
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
use tract_data::prelude::{DatumType, f16};

/// Which activation function a kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ActivationFn {
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
    /// [`ActivationFn::LeakyRelu`].
    MulByScalar,
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
pub enum ActFactory {
    F32(fn() -> Box<dyn ElementWise<f32>>),
    F16(fn() -> Box<dyn ElementWise<f16>>),
    F32Param(fn() -> Box<dyn ElementWise<f32, f32>>),
    F16Param(fn() -> Box<dyn ElementWise<f16, f16>>),
}

/// One registered activation kernel. Submitted via [`inventory`] next to its kernel.
pub struct ActivationImpl {
    pub func: ActivationFn,
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
    pub factory: Option<ActFactory>,
}

impl ActivationImpl {
    /// Runnable on this host right now: compiled in *and* its feature probe passes.
    pub fn is_bound(&self) -> bool {
        self.factory.is_some() && (self.check)()
    }
}

inventory::collect!(ActivationImpl);

/// Every registered descriptor (bound, described, across all compiled-in targets).
pub fn all() -> impl Iterator<Item = &'static ActivationImpl> {
    inventory::iter::<ActivationImpl>()
}

/// The kernel the engine would dispatch to for `(func, dt)` on this host: the
/// highest-`(tier, isa_rank)` descriptor that is bound. `None` if nothing is bound
/// (should never happen while the generic tier is compiled in).
pub fn pick(func: ActivationFn, dt: DatumType) -> Option<&'static ActivationImpl> {
    all()
        .filter(|a| a.func == func && a.dt == dt && a.is_bound())
        .max_by_key(|a| (a.tier, a.isa_rank))
}

/// The `f32` kernel to run for `func` on this host. The generic tier guarantees a
/// bound kernel, so this is infallible for every registered `f32` activation.
pub fn kernel_f32(func: ActivationFn) -> Box<dyn ElementWise<f32>> {
    match pick(func, DatumType::F32).and_then(|a| a.factory) {
        Some(ActFactory::F32(make)) => make(),
        _ => unreachable!("no bound f32 kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f16` kernel to run for `func` on this host. Infallible for every `func`
/// that has an `f16` form registered (all but [`ActivationFn::Erf`]).
pub fn kernel_f16(func: ActivationFn) -> Box<dyn ElementWise<f16>> {
    match pick(func, DatumType::F16).and_then(|a| a.factory) {
        Some(ActFactory::F16(make)) => make(),
        _ => unreachable!("no bound f16 kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f32` kernel for a parameterized activation (e.g. [`ActivationFn::LeakyRelu`]),
/// run with `run_with_params(xs, alpha)`.
pub fn kernel_f32_param(func: ActivationFn) -> Box<dyn ElementWise<f32, f32>> {
    match pick(func, DatumType::F32).and_then(|a| a.factory) {
        Some(ActFactory::F32Param(make)) => make(),
        _ => unreachable!("no bound f32 param kernel for {func:?} (generic tier missing?)"),
    }
}

/// The `f16` kernel for a parameterized activation, run with
/// `run_with_params(xs, alpha)`.
pub fn kernel_f16_param(func: ActivationFn) -> Box<dyn ElementWise<f16, f16>> {
    match pick(func, DatumType::F16).and_then(|a| a.factory) {
        Some(ActFactory::F16Param(make)) => make(),
        _ => unreachable!("no bound f16 param kernel for {func:?} (generic tier missing?)"),
    }
}

// Descriptors live next to their kernels, one file per backend. They are declared
// from here — an always-compiled module — because the arch modules themselves are
// `#[cfg(target_arch)]`-gated and so absent from a foreign-target build, which would
// hide their descriptors from a host drawing the whole matrix. Each per-arch file is
// compiled when building for its arch OR under `registry-all-targets`; its `factory`
// is filled only in the native case, so a foreign symbol is never named.
#[path = "generic/activation.rs"]
mod generic_activation;

#[cfg(any(target_arch = "aarch64", feature = "registry-all-targets"))]
#[path = "arm64/activation.rs"]
mod arm64_activation;

#[cfg(any(target_arch = "arm", feature = "registry-all-targets"))]
#[path = "arm32/activation.rs"]
mod arm32_activation;

#[cfg(any(target_arch = "x86_64", feature = "registry-all-targets"))]
#[path = "x86_64_fma/activation.rs"]
mod x86_64_activation;

#[cfg(any(target_family = "wasm", feature = "registry-all-targets"))]
#[path = "wasm/activation.rs"]
mod wasm_activation;

#[cfg(test)]
mod test {
    use super::*;

    /// Every registered activation resolves to a bound kernel on this host — the
    /// generic tier is the floor, so dispatch can never come up empty.
    #[test]
    fn every_activation_dispatches() {
        crate::setup_test_logger();
        let f32s = [
            ActivationFn::Sigmoid,
            ActivationFn::Silu,
            ActivationFn::Tanh,
            ActivationFn::Erf,
            ActivationFn::HardSwish,
            ActivationFn::Gelu,
        ];
        for func in f32s {
            assert!(pick(func, DatumType::F32).is_some(), "no bound f32 kernel for {func:?}");
        }
        // Erf has no f16 form.
        for func in [
            ActivationFn::Sigmoid,
            ActivationFn::Silu,
            ActivationFn::Tanh,
            ActivationFn::HardSwish,
            ActivationFn::Gelu,
            ActivationFn::LeakyRelu,
            ActivationFn::MulByScalar,
        ] {
            assert!(pick(func, DatumType::F16).is_some(), "no bound f16 kernel for {func:?}");
        }
        for func in [ActivationFn::LeakyRelu, ActivationFn::MulByScalar] {
            assert!(pick(func, DatumType::F32).is_some(), "no bound f32 kernel for {func:?}");
        }
    }

    /// The picked leaky-relu kernel applies `alpha` on the negative side.
    #[test]
    fn leaky_kernel_runs() {
        let op = kernel_f32_param(ActivationFn::LeakyRelu);
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
        let d = pick(ActivationFn::Sigmoid, DatumType::F32).unwrap();
        let Some(ActFactory::F32(f)) = d.factory else { panic!("f32 factory") };
        let mut xs = vec![0f32, 1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 2.0];
        f().run(&mut xs).unwrap();
        for (x, y) in [0f32, 1.0, -1.0, 10.0, -10.0, 0.5, -0.5, 2.0].iter().zip(&xs) {
            let want = 1.0 / (1.0 + (-x).exp());
            assert!((want - y).abs() < 1e-3, "sigmoid({x})={y} want {want}");
        }
    }
}

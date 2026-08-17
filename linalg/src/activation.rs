//! PROTOTYPE — single-source-of-truth registry for element-wise activation kernels.
//!
//! Each concrete kernel submits one [`ActivationImpl`] descriptor through
//! [`inventory`]. The descriptor is pure data (function, dtype, target, tier,
//! kernel name, feature probe) plus an optional `factory`. The factory is `Some`
//! only in a build that targets the kernel's own architecture; a build that merely
//! *describes* a foreign architecture (see the `registry-all-targets` feature)
//! carries the descriptor with `factory: None`, so no foreign symbol is ever named.
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
    // Tanh, Silu, Gelu, Erf, HardSwish, LeakyRelu — added as they are ported.
}

/// How directly a kernel implements its function, best first. The discriminant is
/// the primary selection key: a `Native` kernel always beats a `ViaF32` fallback,
/// which beats the portable `Generic` reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// Portable reference implementation; available on every target.
    Generic = 0,
    /// Reachable but indirect — e.g. f16 computed by round-tripping through an f32
    /// kernel, or `silu` built from `sigmoid`. A dedicated kernel may beat it.
    ViaF32 = 1,
    /// Hand-written kernel for exactly this (function, dtype, target).
    Native = 2,
}

/// A boxed factory for the concrete kernel, monomorphized per element type.
/// Present only when the descriptor's target matches the compile target.
#[derive(Clone, Copy)]
pub enum ActFactory {
    F32(fn() -> Box<dyn ElementWise<f32>>),
    F16(fn() -> Box<dyn ElementWise<f16>>),
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

// ---------------------------------------------------------------------------
// Sigmoid descriptors.
//
// Generic is always compiled. Each arch block is compiled when we build *for*
// that arch OR when `registry-all-targets` is on (so a dev/CI host can draw the
// whole matrix); the `factory` is filled only in the native case, guarded by a
// second cfg so a foreign symbol is never referenced.
// ---------------------------------------------------------------------------

use crate::frame::element_wise::ElementWiseKer;

inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "SSigmoid4",
        check: || true,
        factory: Some(ActFactory::F32(|| crate::generic::SSigmoid4::ew())),
    }
}
inventory::submit! {
    ActivationImpl {
        func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "generic",
        feature: None, tier: Tier::Generic, isa_rank: 0, kernel: "HSigmoid8",
        check: || true,
        factory: Some(ActFactory::F16(|| crate::generic::HSigmoid8::ew())),
    }
}

// -------- aarch64 --------
#[cfg(any(target_arch = "aarch64", feature = "registry-all-targets"))]
mod aarch64_descriptors {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    macro_rules! aarch64_factory {
        (F32, $k:path) => {
            Some(ActFactory::F32(|| <$k>::ew()))
        };
        (F16, $k:path) => {
            Some(ActFactory::F16(|| <$k>::ew()))
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    macro_rules! aarch64_factory {
        ($dt:ident, $k:path) => {
            None
        };
    }
    #[cfg(target_arch = "aarch64")]
    macro_rules! aarch64_check {
        ($e:expr) => {
            $e
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    macro_rules! aarch64_check {
        ($e:expr) => {
            false
        };
    }

    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "aarch64",
            feature: None, tier: Tier::Native, isa_rank: 10,
            kernel: "arm64simd_sigmoid_f32_4n",
            check: || aarch64_check!(true),
            factory: aarch64_factory!(F32, crate::arm64::arm64simd_sigmoid_f32_4n),
        }
    }
    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "aarch64",
            feature: Some("fp16"), tier: Tier::Native, isa_rank: 20,
            kernel: "arm64fp16_sigmoid_f16_8n",
            check: || aarch64_check!(crate::arm64::has_fp16()),
            factory: aarch64_factory!(F16, crate::arm64::arm64fp16_sigmoid_f16_8n),
        }
    }
    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "aarch64",
            feature: None, tier: Tier::ViaF32, isa_rank: 10,
            kernel: "arm64simd_sigmoid_f16_4n",
            check: || aarch64_check!(!crate::arm64::has_fp16()),
            factory: aarch64_factory!(F16, crate::arm64::arm64simd_sigmoid_f16_4n),
        }
    }
}

// -------- x86_64 --------
#[cfg(any(target_arch = "x86_64", feature = "registry-all-targets"))]
mod x86_64_descriptors {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    macro_rules! x86_factory {
        (F32, $k:path) => {
            Some(ActFactory::F32(|| <$k>::ew()))
        };
        (F16, $k:path) => {
            Some(ActFactory::F16(|| <$k>::ew()))
        };
    }
    #[cfg(not(target_arch = "x86_64"))]
    macro_rules! x86_factory {
        ($dt:ident, $k:path) => {
            None
        };
    }
    // `is_x86_feature_detected!` needs a literal argument in its own expansion; it
    // rejects one arriving through a macro metavariable. So probe with literal calls
    // here, and stub to `false` off-target.
    #[cfg(target_arch = "x86_64")]
    mod probe {
        pub fn avx() -> bool {
            std::is_x86_feature_detected!("avx")
        }
        pub fn fma() -> bool {
            std::is_x86_feature_detected!("fma")
        }
        pub fn avx512f() -> bool {
            std::is_x86_feature_detected!("avx512f")
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    mod probe {
        pub fn avx() -> bool {
            false
        }
        pub fn fma() -> bool {
            false
        }
        pub fn avx512f() -> bool {
            false
        }
    }

    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "x86_64",
            feature: Some("avx"), tier: Tier::Native, isa_rank: 10,
            kernel: "avx_sigmoid_f32",
            check: || probe::avx(),
            factory: x86_factory!(F32, crate::x86_64_fma::avx_sigmoid_f32),
        }
    }
    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "x86_64",
            feature: Some("fma"), tier: Tier::Native, isa_rank: 20,
            kernel: "fma_sigmoid_f32",
            check: || probe::fma(),
            factory: x86_factory!(F32, crate::x86_64_fma::fma_sigmoid_f32),
        }
    }
    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "x86_64",
            feature: Some("avx512f"), tier: Tier::Native, isa_rank: 30,
            kernel: "avx512_sigmoid_f32",
            check: || probe::avx512f(),
            factory: x86_factory!(F32, crate::x86_64_fma::avx512_sigmoid_f32),
        }
    }
    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F16, target: "x86_64",
            feature: Some("avx512f"), tier: Tier::ViaF32, isa_rank: 30,
            kernel: "x86_64_avx512_sigmoid_f16_16n",
            check: || probe::avx512f(),
            factory: x86_factory!(F16, crate::x86_64_fma::act_f16::x86_64_avx512_sigmoid_f16_16n),
        }
    }
}

// -------- armv7 (target_arch = "arm") --------
#[cfg(any(target_arch = "arm", feature = "registry-all-targets"))]
mod armv7_descriptors {
    use super::*;

    #[cfg(target_arch = "arm")]
    macro_rules! armv7_factory {
        (F32, $k:path) => {
            Some(ActFactory::F32(|| <$k>::ew()))
        };
    }
    #[cfg(not(target_arch = "arm"))]
    macro_rules! armv7_factory {
        ($dt:ident, $k:path) => {
            None
        };
    }
    #[cfg(target_arch = "arm")]
    macro_rules! armv7_check {
        ($e:expr) => {
            $e
        };
    }
    #[cfg(not(target_arch = "arm"))]
    macro_rules! armv7_check {
        ($e:expr) => {
            false
        };
    }

    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "armv7",
            feature: Some("neon"), tier: Tier::Native, isa_rank: 10,
            kernel: "armv7neon_sigmoid_f32_4n",
            check: || armv7_check!(crate::arm32::has_neon()),
            factory: armv7_factory!(F32, crate::arm32::armv7neon::armv7neon_sigmoid_f32_4n),
        }
    }
}

// -------- wasm --------
// The native sigmoid is a relaxed-simd kernel, and is only compiled into a wasm
// build with `+relaxed-simd`. Feature detection on wasm is compile-time, so the
// runtime `check` collapses to a `cfg!`.
#[cfg(any(target_family = "wasm", feature = "registry-all-targets"))]
mod wasm_descriptors {
    use super::*;

    #[cfg(all(target_family = "wasm", target_feature = "relaxed-simd"))]
    macro_rules! wasm_relaxed_factory {
        () => {
            Some(ActFactory::F32(|| crate::wasm::WasmSigmoid4Relaxed::ew()))
        };
    }
    #[cfg(not(all(target_family = "wasm", target_feature = "relaxed-simd")))]
    macro_rules! wasm_relaxed_factory {
        () => {
            None
        };
    }

    inventory::submit! {
        ActivationImpl {
            func: ActivationFn::Sigmoid, dt: DatumType::F32, target: "wasm",
            feature: Some("relaxed-simd"), tier: Tier::Native, isa_rank: 10,
            kernel: "WasmSigmoid4Relaxed",
            check: || cfg!(all(target_family = "wasm", target_feature = "relaxed-simd")),
            factory: wasm_relaxed_factory!(),
        }
    }
}

/// Render the (function × target) matrix as text. One row per `(func, dtype)`,
/// one column per target seen. Each cell shows the best tier for that cell and a
/// state glyph: `●` bound (runnable here), `○` described (compiled, not runnable
/// here), `·` absent.
pub fn matrix() -> String {
    use std::collections::BTreeSet;
    let descs: Vec<_> = all().collect();

    let targets: Vec<&str> =
        descs.iter().map(|d| d.target).collect::<BTreeSet<_>>().into_iter().collect();
    let rows: Vec<(ActivationFn, DatumType)> =
        descs.iter().map(|d| (d.func, d.dt)).collect::<BTreeSet<_>>().into_iter().collect();

    let tier_glyph = |t: Tier| match t {
        Tier::Native => "N",
        Tier::ViaF32 => "F",
        Tier::Generic => "G",
    };

    let mut out = String::new();
    out.push_str(&format!("{:<16}", "func/dtype"));
    for t in &targets {
        out.push_str(&format!("{:>16}", t));
    }
    out.push('\n');

    for (func, dt) in &rows {
        out.push_str(&format!("{:<16}", format!("{func:?}/{dt:?}")));
        for t in &targets {
            let best = descs
                .iter()
                .filter(|d| d.func == *func && d.dt == *dt && d.target == *t)
                .max_by_key(|d| (d.tier, d.isa_rank));
            let cell = match best {
                None => "·".to_string(),
                Some(d) => {
                    let glyph = if d.is_bound() {
                        "●"
                    } else if d.factory.is_some() {
                        "○" // compiled here but feature absent on this host
                    } else {
                        "○" // described (foreign arch)
                    };
                    format!("{glyph} {}", tier_glyph(d.tier))
                }
            };
            out.push_str(&format!("{cell:>16}"));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    /// The registry must dispatch to the same kernel the legacy `Ops` field does.
    /// Guards the single-source-of-truth migration: registry selection == `plug()`.
    #[test]
    fn matches_legacy_dispatch() {
        crate::setup_test_logger();
        let legacy_f32 = (crate::ops().sigmoid_f32)().name();
        let picked_f32 = pick(ActivationFn::Sigmoid, DatumType::F32).expect("a bound f32 sigmoid");
        assert_eq!(
            picked_f32.kernel, legacy_f32,
            "registry picked {} but plug() dispatches {legacy_f32}",
            picked_f32.kernel
        );

        let legacy_f16 = (crate::ops().sigmoid_f16)().name();
        let picked_f16 = pick(ActivationFn::Sigmoid, DatumType::F16).expect("a bound f16 sigmoid");
        assert_eq!(picked_f16.kernel, legacy_f16);
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

    #[test]
    fn print_matrix() {
        crate::setup_test_logger();
        println!("\n{}", matrix());
    }
}

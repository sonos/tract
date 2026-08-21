//! Cross-arch introspection registry for mmm kernels.
//!
//! Every `MMMExternKernel2!` invocation submits one [`MmmRoutine`] handle to the `inventory`
//! collection, regardless of the build's target arch, so the full function × target matrix
//! is enumerable on any host. The handle carries only what the kernel object cannot: which
//! arch it belongs to, and whether this build actually assembled it. All other metadata
//! (name, tile, quality, datum type, support) is read from the [`MatMatMul`] that `make`
//! builds — the single source of truth — so nothing here duplicates `DynKernel`.
//!
//! On a foreign-arch build the extern symbol is replaced by a bail stub; such a routine has
//! `bound == false`, and its `make()` object is metadata-only and must never be executed.
use crate::mmm::MatMatMul;

/// One mmm kernel, enumerable uniformly on every host.
pub struct MmmRoutine {
    /// Target arch the kernel is written for (e.g. `"arm"`).
    pub target: &'static str,
    /// `true` when this build assembled the kernel (native arch); `false` when `make`
    /// yields a bail stub (described-only, foreign arch).
    pub bound: bool,
    /// Builds the type-erased kernel. Reading its metadata is always safe; *running* it is
    /// only safe when [`Self::bound`] is `true`.
    pub make: fn() -> Box<dyn MatMatMul>,
}

inventory::collect!(MmmRoutine);

/// All registered mmm routines across every compiled-in target.
pub fn all() -> impl Iterator<Item = &'static MmmRoutine> {
    inventory::iter::<MmmRoutine>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_mmm_routines_registered_everywhere() {
        // (target, minimum routines present on any host). arm32: 12 armv7neon + 1 armvfpv2.
        // aarch64: 20 arm64simd + 8 arm64fp16. x86_64: 7 avx + 8 fma + 9 avx512 + 2 i32.
        // sve, sme, apple amx, dotprod, vnni and amx ride on os and assembler-probe cfgs.
        for (target, min) in [("arm", 13), ("aarch64", 28), ("x86_64", 26)] {
            let routines: Vec<_> = all().filter(|r| r.target == target).collect();
            assert!(routines.len() >= min, "{target} mmm routines: {} < {min}", routines.len());
            // bound iff this build's arch is the routine's target (else it's a bail stub).
            let bound_expected = target == std::env::consts::ARCH;
            for r in &routines {
                // Metadata is readable on any host (the stub only bails when *run*).
                let ker = (r.make)();
                assert_eq!(r.bound, bound_expected, "{target}/{} bound flag", ker.name());
            }
        }
    }
}

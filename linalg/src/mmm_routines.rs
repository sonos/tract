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
    fn arm32_routines_are_registered_everywhere() {
        let arm32: Vec<_> = all().filter(|r| r.target == "arm").collect();
        // 12 armv7neon + 1 armvfpv2.
        assert_eq!(arm32.len(), 13, "arm32 mmm routines: {}", arm32.len());
        let bound_expected = cfg!(target_arch = "arm");
        for r in &arm32 {
            // Metadata is readable on any host (the stub only bails when *run*).
            let ker = (r.make)();
            assert_eq!(r.bound, bound_expected, "{} bound flag", ker.name());
        }
    }
}

//! Cross-arch introspection registry for mmm kernels.
//!
//! Every arch-prefixed `MMMExternKernel!` / `MMMRustKernel!` invocation submits one
//! [`MmmRoutine`] handle to the `inventory` collection, so every kernel tree this build
//! compiled is enumerable: the full function × target matrix under the `foreign-inventory`
//! feature, this arch's own share without it. The handle carries only what the kernel object
//! cannot: which arch it belongs to, and whether this build actually assembled it. All other
//! metadata (name, tile, quality, datum type, support) is read from the [`MatMatMul`] that
//! `make` builds — the single source of truth — so nothing here duplicates `DynKernel`.
//!
//! A foreign tree's kernels are bail stubs; such routines have `bound == false`, and their
//! `make()` object is metadata-only and must never be executed.
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

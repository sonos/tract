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
//!
//! Enumeration is complete across *architectures* but conditional on the *toolchain*: a kernel
//! whose asm this build could not assemble at all — SVE and SME behind their build.rs assembler
//! probes, the fp16 tree under `no_fp16` — is not declared here, so its absence means "this
//! toolchain cannot build it", not "no such kernel exists".
use crate::mmm::MatMatMul;

/// One mmm kernel, enumerable uniformly on every host.
pub struct MmmRoutine {
    /// Arch the kernel is written for, or `None` when it is portable Rust that
    /// every target builds.
    pub target: Option<crate::platform::Target>,
    /// Builds the type-erased kernel. Reading its metadata is always safe; *running* it is
    /// only safe when [`Self::bound`] is `true`.
    pub make: fn() -> Box<dyn MatMatMul>,
}

impl MmmRoutine {
    /// Whether this build assembled the kernel: a portable one always, an arch's only when
    /// building for it. Elsewhere `make` yields a bail stub, good for its metadata alone.
    pub fn bound(&self) -> bool {
        self.target.is_none_or(|t| t.is_native())
    }
}

inventory::collect!(MmmRoutine);

/// All registered mmm routines across every compiled-in target.
pub fn all() -> impl Iterator<Item = &'static MmmRoutine> {
    inventory::iter::<MmmRoutine>()
}

/// The pool as `target` would see it: that target's kernels plus the portable ones, whether
/// or not this build assembled them. Support predicates still answer for the running host,
/// so kernels behind a hardware feature (fp16, dotprod, sve) are kept rather than filtered —
/// this is for inspecting another platform's dispatch, and anything it returns unbound will
/// panic if actually called.
pub fn pool_for(target: crate::platform::Target) -> Vec<Box<dyn MatMatMul>> {
    let mut pool: Vec<Box<dyn MatMatMul>> = all()
        .filter(|r| r.target.is_none_or(|t| t == target))
        .filter_map(|r| {
            let kernel = (r.make)();
            // A kernel this build assembled can answer its own support predicate, and the
            // pool it belongs to honours it; one it did not assemble cannot, so it stays.
            (!r.bound() || kernel.is_supported_here()).then_some(kernel)
        })
        .collect();
    pool.sort_by(|a, b| a.name().cmp(b.name()));
    pool
}

/// Every kernel this machine can run: what the dispatch pool holds, whichever selection
/// policy is plugged over it. Sorted by name, because `inventory` yields link order, which
/// is not stable across builds, while pool position still breaks ties in selection.
pub fn pool() -> Vec<Box<dyn MatMatMul>> {
    let mut pool: Vec<Box<dyn MatMatMul>> =
        all().filter(|r| r.bound()).map(|r| (r.make)()).filter(|k| k.is_supported_here()).collect();
    pool.sort_by(|a, b| a.name().cmp(b.name()));
    pool
}

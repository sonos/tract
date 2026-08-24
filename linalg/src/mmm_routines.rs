//! Cross-arch introspection registry for mmm kernels.
//!
//! Every arch-prefixed `MMMExternKernel!` / `MMMRustKernel!` invocation submits one
//! [`MmmRoutine`] handle to the `inventory` collection, so every kernel tree this build
//! compiled is enumerable: the full function × target matrix under the `foreign-inventory`
//! feature, this arch's own share without it. The handle carries only what the kernel object
//! cannot: which arch it belongs to. Everything else — name, tile, quality, datum type, whether
//! this build compiled it, whether it runs here — is read from the [`MatMatMul`] that `make`
//! builds, so nothing here duplicates `DynKernel`.
//!
//! A foreign tree's kernels are bail stubs; they answer [`MatMatMul::built`] with false, and
//! their `make()` object is metadata-only and must never be executed.
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
    pub target: Option<crate::isa::Arch>,
    /// Builds the type-erased kernel. Reading its metadata is always safe; *running* it is
    /// only safe when the kernel answers [`MatMatMul::built`] with true.
    pub make: fn() -> Box<dyn MatMatMul>,
}

inventory::collect!(MmmRoutine);

/// All registered mmm routines across every compiled-in target.
pub fn declared() -> impl Iterator<Item = &'static MmmRoutine> {
    inventory::iter::<MmmRoutine>()
}

/// The runnable set as `target` would see it: that target’s kernels plus the portable ones, whether
/// or not this build assembled them. Instruction-set requirements are still answered against
/// the running host, so reaching a cohort behind a feature this host lacks (fp16, dotprod,
/// sve2) means adding it with `TRACT_CPU_ISA` — and anything this returns unbuilt will panic
/// if actually called.
pub fn runnable_for(target: crate::isa::Arch) -> Vec<Box<dyn MatMatMul>> {
    let mut pool: Vec<Box<dyn MatMatMul>> = declared()
        .filter(|r| r.target.is_none_or(|t| t == target))
        .map(|r| (r.make)())
        // A kernel this build compiled has to be runnable here to be in the set; one it did
        // not compile cannot answer that, so it stays as metadata.
        .filter(|kernel| !kernel.built() || kernel.runnable())
        .collect();
    pool.sort_by(|a, b| a.name().cmp(b.name()));
    pool
}

/// Every kernel this machine can run: what dispatch chooses from, whichever selection
/// policy is plugged over it. Sorted by name, because `inventory` yields link order, which
/// is not stable across builds, while position in the set still breaks ties in selection.
pub fn runnable() -> Vec<Box<dyn MatMatMul>> {
    let mut pool: Vec<Box<dyn MatMatMul>> =
        declared().map(|r| (r.make)()).filter(|k| k.runnable()).collect();
    pool.sort_by(|a, b| a.name().cmp(b.name()));
    pool
}

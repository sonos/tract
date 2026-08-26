//! Cross-arch introspection registry for mmm kernels.
//!
//! Every arch-prefixed `MMMExternKernel!` / `MMMRustKernel!` invocation submits one
//! [`MmmRoutine`] handle to the `inventory` collection, so every kernel tree this build
//! compiled is enumerable: the full function × target matrix under the `foreign-inventory`
//! feature, this arch's own share without it. The handle carries nothing but the constructor:
//! arch, name, tile, datum type, whether this build compiled it and whether it runs here are
//! all read from the [`MatMatMul`] that `make` builds, so nothing here duplicates `DynKernel`.
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
    /// Builds the type-erased kernel. Reading its metadata is always safe; *running* it is
    /// only safe when the kernel answers [`MatMatMul::built`] with true.
    pub make: fn() -> Box<dyn MatMatMul>,
}

inventory::collect!(MmmRoutine);

/// One panel extractor, enumerable uniformly on every host. Unlike [`MmmRoutine`] the handle does
/// carry its arch, an extractor being only ever called and never enumerated for its metadata, so
/// nothing builds one to ask. Whether this build compiled its body, and what the instruction set
/// must offer, are fields of the [`PanelExtractor`] itself.
pub struct MmmExtractor {
    pub target: crate::isa::Arch,
    pub make: fn() -> crate::mmm::PanelExtractor,
}

inventory::collect!(MmmExtractor);

/// Every panel extractor this build compiled, whichever architecture it belongs to.
pub fn declared_extractors() -> impl Iterator<Item = &'static MmmExtractor> {
    inventory::iter::<MmmExtractor>()
}

/// The extractors a machine can run: its architecture's, needing nothing its instruction set
/// lacks. An extractor this build did not compile stays out -- unlike a kernel, an extractor is
/// only ever called, never enumerated for its metadata.
pub fn extractors_for(isa: &crate::isa::IsaSet) -> Vec<crate::mmm::PanelExtractor> {
    let Some(arch) = isa.arch() else { return vec![] };
    let mut pool: Vec<crate::mmm::PanelExtractor> = declared_extractors()
        .filter(|e| e.target == arch)
        .map(|e| (e.make)())
        .filter(|e| e.built && e.isa.satisfied_by(*isa))
        .collect();
    pool.sort_by(|a, b| a.name.cmp(&b.name));
    pool
}

/// All registered mmm routines across every compiled-in target.
pub fn declared() -> impl Iterator<Item = &'static MmmRoutine> {
    inventory::iter::<MmmRoutine>()
}

/// The runnable set as `target` would see it: that target’s kernels plus the generic ones, whether
/// or not this build assembled them. Instruction-set requirements are still answered against
/// the running host, so reaching a cohort behind a feature this host lacks (fp16, dotprod,
/// sve2) means adding it with `TRACT_CPU_ISA` — and anything this returns unbuilt will panic
/// if actually called.
pub fn runnable_for(target: crate::isa::Arch) -> Vec<Box<dyn MatMatMul>> {
    let mut pool: Vec<Box<dyn MatMatMul>> = declared()
        .map(|r| (r.make)())
        .filter(|kernel| kernel.arch().is_none_or(|a| a == target))
        // A kernel this build compiled has to be runnable here to be in the set; one it did
        // not compile cannot answer that, so it stays as metadata.
        .filter(|kernel| !kernel.built() || kernel.runnable())
        .collect();
    pool.sort_by(|a, b| a.name().cmp(b.name()));
    pool
}

/// Every kernel this machine can run: what dispatch chooses from, whichever selection
/// policy runs over it. Sorted by name, because `inventory` yields link order, which
/// is not stable across builds, while position in the set still breaks ties in selection.
pub fn runnable() -> Vec<Box<dyn MatMatMul>> {
    let mut pool: Vec<Box<dyn MatMatMul>> =
        declared().map(|r| (r.make)()).filter(|k| k.runnable()).collect();
    pool.sort_by(|a, b| a.name().cmp(b.name()));
    pool
}

//! Per-platform dispatch policy, enumerable like the kernels themselves.
//!
//! Each arch tree submits one [`PlatformSelector`]: the function installing the closures that
//! answer "which kernel for this shape" over the pool. Every tree the build compiled
//! contributes one, so the policies are enumerable on any host, while [`Target::is_native`]
//! says which one this build actually runs — the only one [`crate::best`] applies.

use crate::Ops;

/// A platform tract names, after its `target_arch`.
///
/// Naming one is not having kernels for it: [`RiscV64`](Target::RiscV64) has no tree yet, so a
/// riscv64 build is portable-only even though it knows what it is running on, and an
/// architecture tract does not name at all has no variant here. Only the wasm tree hinges on a
/// build feature — hence the variant naming that feature; the others exist on their arch and
/// gate individual kernels on runtime probes instead.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Target {
    Arm,
    Aarch64,
    X86_64,
    RiscV64,
    /// The only tree gated at build time rather than probed at runtime.
    Wasm32Simd128,
}

impl Target {
    pub fn is_native(&self) -> bool {
        match self {
            Target::Arm => cfg!(target_arch = "arm"),
            Target::Aarch64 => cfg!(target_arch = "aarch64"),
            Target::X86_64 => cfg!(target_arch = "x86_64"),
            Target::RiscV64 => cfg!(target_arch = "riscv64"),
            Target::Wasm32Simd128 => {
                cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
            }
        }
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            Target::Arm => "arm",
            Target::Aarch64 => "aarch64",
            Target::X86_64 => "x86_64",
            Target::RiscV64 => "riscv64",
            Target::Wasm32Simd128 => "wasm32+simd128",
        };
        write!(f, "{s}")
    }
}

pub struct PlatformSelector {
    /// Arch the policy is written for. Not an `Option`, unlike
    /// [`crate::mmm_routines::MmmRoutine::target`]: a kernel can be portable, a dispatch
    /// policy is always somebody's.
    pub target: Target,
    pub plug: fn(&mut Ops),
}

inventory::collect!(PlatformSelector);

/// Every policy compiled into this build, native or not.
pub fn all() -> impl Iterator<Item = &'static PlatformSelector> {
    inventory::iter::<PlatformSelector>()
}

/// `Ops` as `target` sees them: its kernels, from [`crate::mmm_routines::pool_for`], under its
/// own policy. Answers which kernel that platform would choose for a shape, from any host —
/// what it cannot reproduce is a hardware probe, so a cohort behind one (fp16, dotprod, SVE)
/// needs its `TRACT_CPU_*` knob set to be reached. `None` when the target's tree was not
/// compiled in; see the `foreign-inventory` feature.
pub fn inspect(target: Target) -> Option<Ops> {
    let selector = all().find(|s| s.target == target)?;
    let mut ops = crate::generic();
    ops.mmm_impls = crate::mmm_routines::pool_for(target);
    (selector.plug)(&mut ops);
    Some(ops)
}

//! Per-platform dispatch policy, enumerable like the kernels themselves.
//!
//! Each arch tree submits one [`ArchPlug`]: the function installing the closures that
//! answer "which kernel for this shape" over the runnable set. Every tree the build compiled
//! contributes one, so the policies are enumerable on any host, while [`Arch::is_native`]
//! says which one this build actually runs — the only one [`crate::best`] applies.

use crate::Ops;

/// An architecture tract has a kernel tree for, after its `target_arch`.
///
/// Naming one is not having kernels for it: [`RiscV64`](Arch::RiscV64) has no tree yet, so a
/// riscv64 build is portable-only even though it knows what it is running on, and an
/// architecture tract does not name at all has no variant here. Only the wasm tree hinges on a
/// build feature — hence the variant naming that feature; the others exist on their arch and
/// gate individual kernels on runtime probes instead.
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Arch {
    Arm,
    Aarch64,
    X86_64,
    RiscV64,
    /// The only tree gated at build time rather than probed at runtime.
    Wasm32Simd128,
}

impl Arch {
    pub fn is_native(&self) -> bool {
        match self {
            Arch::Arm => cfg!(target_arch = "arm"),
            Arch::Aarch64 => cfg!(target_arch = "aarch64"),
            Arch::X86_64 => cfg!(target_arch = "x86_64"),
            Arch::RiscV64 => cfg!(target_arch = "riscv64"),
            Arch::Wasm32Simd128 => {
                cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
            }
        }
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            Arch::Arm => "arm",
            Arch::Aarch64 => "aarch64",
            Arch::X86_64 => "x86_64",
            Arch::RiscV64 => "riscv64",
            Arch::Wasm32Simd128 => "wasm32+simd128",
        };
        write!(f, "{s}")
    }
}

pub struct ArchPlug {
    /// Arch the policy is written for. Not an `Option`, unlike
    /// [`crate::mmm_routines::MmmRoutine::target`]: a kernel can be portable, a dispatch
    /// policy is always somebody's.
    pub arch: Arch,
    pub plug: fn(&mut Ops),
}

inventory::collect!(ArchPlug);

/// Every policy compiled into this build, native or not.
pub fn all() -> impl Iterator<Item = &'static ArchPlug> {
    inventory::iter::<ArchPlug>()
}

/// `Ops` as `arch` sees them: its kernels, from [`crate::mmm_routines::runnable_for`], under its
/// own tiers. Answers which kernel that architecture would choose for a shape, from any host.
/// What it cannot reproduce is a hardware probe, so it starts from the plain architecture and
/// nothing else: a cohort behind a feature is reached by naming that feature in `TRACT_CPU_ISA`,
/// which is checked against this architecture rather than the host's. `None` when the
/// architecture's tree was not compiled in; see the `foreign-inventory` feature.
pub fn inspect(arch: Arch) -> Option<Ops> {
    let plug = all().find(|s| s.arch == arch)?;
    let isa = if arch.is_native() {
        crate::isa::native()
    } else {
        crate::isa::forced(crate::isa::IsaSet::of_arch(arch))
    };
    let mut ops = crate::generic_for(isa);
    (plug.plug)(&mut ops);
    Some(ops)
}

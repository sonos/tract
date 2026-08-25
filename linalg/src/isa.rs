//! Instruction-set features, as data a kernel declares and a machine is asked for.
//!
//! A kernel's requirement is a set rather than a closure: it can be printed, enumerated, and
//! evaluated against a machine other than this one, which is what makes another cohort's
//! dispatch inspectable. Micro-architecture is deliberately not in here — how many 512-bit FMA
//! ports a core has is not an instruction set, and preferring a kernel is not the same as being
//! able to run it: that belongs in [`crate::mmm::MatMatMulKer::dynamic_boost`].

use std::fmt;
use std::sync::OnceLock;

/// An architecture tract has a kernel tree for, after its `target_arch`. It is the identity a
/// kernel tree and a dispatch tier are keyed on; [`Isa::of_arch`] is the same thing as a set
/// member, and [`IsaSet::arch`] reads it back out of a machine's features.
///
/// Naming one is not having kernels for it: [`RiscV64`](Arch::RiscV64) has no tree yet, so a
/// riscv64 build is generic-only even though it knows what it is running on, and an
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
    pub const ALL: [Arch; 5] =
        [Arch::Arm, Arch::Aarch64, Arch::X86_64, Arch::RiscV64, Arch::Wasm32Simd128];

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

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Arch::Arm => "arm",
            Arch::Aarch64 => "aarch64",
            Arch::X86_64 => "x86_64",
            Arch::RiscV64 => "riscv64",
            Arch::Wasm32Simd128 => "wasm32+simd128",
        })
    }
}

/// One instruction-set feature a kernel can need, or — at level 0 — the plain architecture
/// underneath them all.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Isa {
    /// The plain architectures. One belongs in every set, and it is what says whose kernels the
    /// set is talking about; two is a contradiction, since no machine implements both.
    Arm,
    Aarch64,
    X86_64,
    RiscV64,
    Wasm32,
    /// A step on armv7 only: on aarch64 Advanced SIMD is baseline, hence unnamed there.
    ArmNeon,
    Aarch64Fp16,
    Aarch64DotProd,
    Aarch64Sve2,
    Aarch64Sme,
    Aarch64Sme2,
    Aarch64AppleAmx,
    X86_64Avx,
    X86_64Avx2,
    X86_64Fma,
    X86_64F16c,
    X86_64Avx512f,
    X86_64Avx512Vnni,
    X86_64Avx512Fp16,
    X86_64AvxVnni,
    X86_64AmxInt8,
    X86_64AmxBf16,
    Wasm32Simd128,
    Wasm32RelaxedSimd,
}

impl Isa {
    pub const ALL: [Isa; 24] = [
        Isa::Arm,
        Isa::Aarch64,
        Isa::X86_64,
        Isa::RiscV64,
        Isa::Wasm32,
        Isa::ArmNeon,
        Isa::Aarch64Fp16,
        Isa::Aarch64DotProd,
        Isa::Aarch64Sve2,
        Isa::Aarch64Sme,
        Isa::Aarch64Sme2,
        Isa::Aarch64AppleAmx,
        Isa::X86_64Avx,
        Isa::X86_64Avx2,
        Isa::X86_64Fma,
        Isa::X86_64F16c,
        Isa::X86_64Avx512f,
        Isa::X86_64Avx512Vnni,
        Isa::X86_64Avx512Fp16,
        Isa::X86_64AvxVnni,
        Isa::X86_64AmxInt8,
        Isa::X86_64AmxBf16,
        Isa::Wasm32Simd128,
        Isa::Wasm32RelaxedSimd,
    ];

    /// The token as it appears in a report and in `TRACT_CPU_ISA`.
    pub fn name(&self) -> &'static str {
        match self {
            Isa::Arm => "arm",
            Isa::Aarch64 => "aarch64",
            Isa::X86_64 => "x86_64",
            Isa::RiscV64 => "riscv64",
            Isa::Wasm32 => "wasm32",
            Isa::ArmNeon => "neon",
            Isa::Aarch64Fp16 => "fp16",
            Isa::Aarch64DotProd => "dotprod",
            Isa::Aarch64Sve2 => "sve2",
            Isa::Aarch64Sme => "sme",
            Isa::Aarch64Sme2 => "sme2",
            Isa::Aarch64AppleAmx => "apple-amx",
            Isa::X86_64Avx => "avx",
            Isa::X86_64Avx2 => "avx2",
            Isa::X86_64Fma => "fma",
            Isa::X86_64F16c => "f16c",
            Isa::X86_64Avx512f => "avx512f",
            Isa::X86_64Avx512Vnni => "avx512vnni",
            Isa::X86_64Avx512Fp16 => "avx512fp16",
            Isa::X86_64AvxVnni => "avxvnni",
            Isa::X86_64AmxInt8 => "amx-int8",
            Isa::X86_64AmxBf16 => "amx-bf16",
            Isa::Wasm32Simd128 => "simd128",
            Isa::Wasm32RelaxedSimd => "relaxed-simd",
        }
    }

    fn from_name(s: &str) -> Option<Isa> {
        Isa::ALL.into_iter().find(|i| i.name() == s)
    }

    /// Where this feature sits in its architecture's ladder, each step meaning "a kernel
    /// written for this needs nothing a kernel written for the step below has, and can do
    /// more". Steps are compared across an architecture's whole kernel set, so every feature
    /// a kernel can declare has to be placed: an unplaced feature reads as the baseline and
    /// would quietly demote its kernels below every sibling in the preference order.
    ///
    /// The two architectures share the scale without meeting on it — no host offers features
    /// from both — so `Neon` and `Avx` both sitting at 1 says nothing about each other.
    /// Widening the scale means revisiting [`MAX_LEVEL`].
    ///
    /// This is tract's own ladder, one step per feature it dispatches on; it is not the psABI's
    /// `x86-64-v1..v4`, which bundles features into four named levels. Only the word is
    /// borrowed, never the numbering.
    pub const fn level(&self) -> u8 {
        match self {
            // The plain architecture is the floor every ladder rises from.
            Isa::Arm | Isa::Aarch64 | Isa::X86_64 | Isa::RiscV64 | Isa::Wasm32 => 0,
            // x86: each generation subsumes the last, AMX above the VNNI it needs alongside it.
            Isa::X86_64Avx => 1,
            Isa::X86_64Avx2 | Isa::X86_64Fma | Isa::X86_64F16c => 2,
            Isa::X86_64Avx512f | Isa::X86_64AvxVnni => 3,
            Isa::X86_64Avx512Vnni => 4,
            // A native-f16 kernel beats the f32 round-trip a plain AVX-512 core is left with,
            // so this sits a step above the set it extends.
            Isa::X86_64Avx512Fp16 => 4,
            Isa::X86_64AmxInt8 | Isa::X86_64AmxBf16 => 5,
            // arm: NEON is the armv7 step above bare VFP, and baseline on aarch64 where the
            // ladder continues through the matrix extensions.
            Isa::ArmNeon => 1,
            Isa::Aarch64Fp16 | Isa::Aarch64DotProd => 2,
            Isa::Aarch64Sve2 => 3,
            Isa::Aarch64Sme | Isa::Aarch64Sme2 => 4,
            Isa::Aarch64AppleAmx => 5,
            // relaxed-simd brings the fused multiply-add the baseline proposal lacks.
            Isa::Wasm32Simd128 => 0,
            Isa::Wasm32RelaxedSimd => 1,
        }
    }

    /// Whose instruction set this belongs to. Every feature belongs to exactly one architecture —
    /// that is what makes a set holding two of them a contradiction rather than a rich machine,
    /// and what lets `TRACT_CPU_ISA` reject a feature the architecture cannot have.
    pub const fn arch(&self) -> Arch {
        match self {
            Isa::Arm | Isa::ArmNeon => Arch::Arm,
            Isa::Aarch64
            | Isa::Aarch64Fp16
            | Isa::Aarch64DotProd
            | Isa::Aarch64Sve2
            | Isa::Aarch64Sme
            | Isa::Aarch64Sme2
            | Isa::Aarch64AppleAmx => Arch::Aarch64,
            Isa::X86_64
            | Isa::X86_64Avx
            | Isa::X86_64Avx2
            | Isa::X86_64Fma
            | Isa::X86_64F16c
            | Isa::X86_64Avx512f
            | Isa::X86_64Avx512Vnni
            | Isa::X86_64Avx512Fp16
            | Isa::X86_64AvxVnni
            | Isa::X86_64AmxInt8
            | Isa::X86_64AmxBf16 => Arch::X86_64,
            Isa::RiscV64 => Arch::RiscV64,
            Isa::Wasm32 | Isa::Wasm32Simd128 | Isa::Wasm32RelaxedSimd => Arch::Wasm32Simd128,
        }
    }

    /// Whether this is a plain architecture rather than a feature on top of one.
    pub const fn is_arch(&self) -> bool {
        matches!(self, Isa::Arm | Isa::Aarch64 | Isa::X86_64 | Isa::RiscV64 | Isa::Wasm32)
    }

    /// The set member that stands for `arch` itself.
    pub const fn of_arch(arch: Arch) -> Isa {
        match arch {
            Arch::Arm => Isa::Arm,
            Arch::Aarch64 => Isa::Aarch64,
            Arch::X86_64 => Isa::X86_64,
            Arch::RiscV64 => Isa::RiscV64,
            Arch::Wasm32Simd128 => Isa::Wasm32,
        }
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What one machine offers: the architecture it is, and the features it adds on top. A set is
/// well formed when it holds exactly one architecture — [`IsaSet::of_arch`] is how to start one,
/// and [`IsaSet::arch`] reads it back.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct IsaSet(u32);

impl IsaSet {
    pub const fn empty() -> IsaSet {
        IsaSet(0)
    }

    /// The plain architecture, with none of its features yet.
    pub const fn of_arch(arch: Arch) -> IsaSet {
        IsaSet::empty().with(Isa::of_arch(arch))
    }

    /// The architecture this set speaks for, `None` for the empty set. Two architectures in one
    /// set is a contradiction no machine can be, so the first one found wins and the set should
    /// never have been built that way — see [`IsaSet::of_arch`].
    pub fn arch(self) -> Option<Arch> {
        self.iter().find(|i| i.is_arch()).map(|i| i.arch())
    }

    pub const fn with(self, isa: Isa) -> IsaSet {
        IsaSet(self.0 | 1 << isa as u32)
    }

    pub const fn without(self, isa: Isa) -> IsaSet {
        IsaSet(self.0 & !(1 << isa as u32))
    }

    pub const fn has(self, isa: Isa) -> bool {
        self.0 & (1 << isa as u32) != 0
    }

    /// The machine an architecture's ladder reaches at `level`: that architecture, plus every
    /// feature of it at or below the step. Not every real part is one of these -- a feature can
    /// ship without its level-mates -- but these are the generations a kernel set is written
    /// against, so they are what a matrix column, and a test asking "on which machines", mean by
    /// a machine.
    pub fn ladder(arch: Arch, level: u8) -> IsaSet {
        let mut set = IsaSet::of_arch(arch);
        for isa in Isa::ALL {
            if isa.arch() == arch && isa.level() <= level {
                set = set.with(isa);
            }
        }
        set
    }

    /// Every generation of every architecture tract has a kernel tree for, as
    /// [`IsaSet::ladder`] defines one. What a matrix enumerates, and what a question about all
    /// machines at once ranges over.
    pub fn every_ladder() -> impl Iterator<Item = IsaSet> {
        Arch::ALL.into_iter().flat_map(|arch| (0..=MAX_LEVEL).map(move |l| IsaSet::ladder(arch, l)))
    }
    pub fn iter(self) -> impl Iterator<Item = Isa> {
        Isa::ALL.into_iter().filter(move |i| self.has(*i))
    }
}

impl fmt::Debug for IsaSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            return f.write_str("-");
        }
        f.write_str(&self.iter().map(|i| i.name()).collect::<Vec<_>>().join(","))
    }
}

/// What a kernel needs from the instruction set to be able to run at all. Whether it is the
/// *best* thing that can run is a different question, and not this type's business.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct IsaReq {
    /// Every one of these must be present.
    pub needs: &'static [Isa],
}

impl IsaReq {
    /// Runs anywhere its arch does.
    pub const ANY: IsaReq = IsaReq { needs: &[] };

    pub const fn needing(self, needs: &'static [Isa]) -> IsaReq {
        IsaReq { needs }
    }

    pub fn satisfied_by(&self, set: IsaSet) -> bool {
        self.needs.iter().all(|i| set.has(*i))
    }

    /// The most capable lineage step this kernel sits in, feeding the default half of
    /// [`crate::mmm::MatMatMulKer::dynamic_boost`].
    pub fn level(&self) -> u8 {
        self.needs.iter().map(|i| i.level()).max().unwrap_or(0)
    }
}

/// What one step up the instruction-set ladder is worth when nothing else is known. A kernel
/// written against a more capable set is assumed better than one written against a less capable
/// one; this is the size of that assumption, in the same units as a declared `boost`.
///
/// A declared boost is how an exception to that assumption is spelled, so it has to cover the
/// ladder steps it disagrees with -- and only those: a kernel whose competition sits in its own
/// level disagrees with no step and needs no magnitude at all. Spell the ones that do cross levels
/// with [`peer_of`] instead of a literal, so the claim survives a ladder that grows a step;
/// [`NEVER_PREFERRED`] is the far end of the range, for a kernel that must lose every tie.
pub const LEVEL_BOOST: isize = 10;

/// The deepest step any ladder reaches, bounding what a boost has to be able to cross.
pub const MAX_LEVEL: u8 = 5;

/// A boost that cancels the ladder between two steps, for a kernel written for `mine` but
/// measured as a peer of the kernels written for `theirs`. The relation is the claim; the
/// number is derived from it, and stays right when a step is inserted between the two.
pub const fn peer_of(mine: Isa, theirs: Isa) -> isize {
    (theirs.level() as isize - mine.level() as isize) * LEVEL_BOOST
}

/// A boost no level can make up for, for a kernel that is runnable here but must never be
/// chosen unless something outside the preference order asks for it by name.
pub const NEVER_PREFERRED: isize = -(LEVEL_BOOST * MAX_LEVEL as isize) - 1;

impl fmt::Debug for IsaReq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let needs = self.needs.iter().map(|i| i.name()).collect::<Vec<_>>().join("+");
        if needs.is_empty() {
            f.write_str("any")?;
        } else {
            f.write_str(&needs)?;
        }
        Ok(())
    }
}

/// What this machine has: probed once, then edited by `TRACT_CPU_ISA`. Only the running
/// architecture's tree is asked — a foreign tree compiled in for enumeration would be probing
/// this host about features it cannot have.
pub fn native() -> IsaSet {
    static NATIVE: OnceLock<IsaSet> = OnceLock::new();
    *NATIVE.get_or_init(|| {
        let set = forced(probe());
        log::debug!("ISA: {set:?}");
        set
    })
}

fn probe() -> IsaSet {
    #[cfg(target_arch = "arm")]
    return crate::arm32::isa_set();
    #[cfg(target_arch = "aarch64")]
    return crate::arm64::isa_set();
    #[cfg(target_arch = "x86_64")]
    return crate::x86_64::isa_set();
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return crate::wasm::isa_set();
    // An architecture with no kernel tree, or wasm without simd128: nothing to declare, and no
    // architecture to name either, since none of its kernels would be reachable anyway.
    #[cfg(not(any(
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    IsaSet::empty()
}

/// `TRACT_CPU_ISA=+sve2,-fp16` edits the probed set, so one knob covers every feature and a
/// cohort this machine is not can be asked what it would dispatch. Nothing checks that the
/// result is a CPU that could exist: asking for avx512f without fma describes no hardware, and
/// dispatch will answer for it anyway.
/// Apply `TRACT_CPU_ISA` to `set`. A token naming another architecture's feature is a hard error
/// rather than a warning: it cannot do what it asks for, and silently doing nothing has it look
/// like the feature was tried and made no difference. To reason about another architecture, start
/// from its own set — [`IsaSet::of_arch`], which is what [`crate::platform::inspect`] does.
pub(crate) fn forced(mut set: IsaSet) -> IsaSet {
    let Some(spec) = crate::knobs::TRACT_CPU_ISA.get() else { return set };
    for token in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (add, name) = match token.split_at(1) {
            ("+", name) => (true, name),
            ("-", name) => (false, name),
            _ => (true, token),
        };
        let Some(isa) = Isa::from_name(name) else {
            log::warn!("TRACT_CPU_ISA: unknown feature {name:?}, ignored");
            continue;
        };
        if let Some(arch) = set.arch() {
            assert!(
                isa.arch() == arch,
                "TRACT_CPU_ISA: {name} belongs to {}, and this set is {arch} — a machine is one \
                 architecture, so the token cannot apply",
                isa.arch()
            );
        }
        set = if add { set.with(isa) } else { set.without(isa) };
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MAX_LEVEL` bounds what a declared boost has to be able to cross, so a ladder step added
    /// beyond it would make [`NEVER_PREFERRED`] and every [`peer_of`] claim too small.
    #[test]
    fn ladder_stays_within_the_bound() {
        for isa in Isa::ALL {
            assert!(
                isa.level() <= MAX_LEVEL,
                "{isa} is level {}, past MAX_LEVEL {MAX_LEVEL}",
                isa.level()
            );
        }
    }

    #[test]
    fn peer_of_cancels_the_steps_between() {
        assert_eq!(peer_of(Isa::X86_64Fma, Isa::X86_64Avx512f), LEVEL_BOOST);
        assert_eq!(peer_of(Isa::X86_64Avx, Isa::X86_64Avx512Vnni), 3 * LEVEL_BOOST);
        assert_eq!(peer_of(Isa::X86_64Avx512f, Isa::X86_64AvxVnni), 0);
    }
}

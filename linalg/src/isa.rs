//! Instruction-set features, as data a kernel declares and a machine is asked for.
//!
//! A kernel's requirement is a set rather than a closure: it can be printed, enumerated, and
//! evaluated against a machine other than this one, which is what makes another cohort's
//! dispatch inspectable. Micro-architecture is deliberately not in here — how many 512-bit FMA
//! ports a core has is not an instruction set; see [`IsaReq::probe`].

use std::fmt;
use std::sync::OnceLock;

/// One instruction-set feature a kernel can need.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Isa {
    /// armv7. On aarch64 NEON is baseline, hence unnamed there.
    Neon,
    Fp16,
    DotProd,
    Sve2,
    Sme,
    Sme2,
    /// Apple's AMX coprocessor, which is not the x86 extension of the same name.
    AppleAmx,
    Avx,
    Avx2,
    Fma,
    F16c,
    Avx512f,
    Avx512Vnni,
    AvxVnni,
    AmxInt8,
    AmxBf16,
    Simd128,
}

impl Isa {
    pub const ALL: [Isa; 17] = [
        Isa::Neon,
        Isa::Fp16,
        Isa::DotProd,
        Isa::Sve2,
        Isa::Sme,
        Isa::Sme2,
        Isa::AppleAmx,
        Isa::Avx,
        Isa::Avx2,
        Isa::Fma,
        Isa::F16c,
        Isa::Avx512f,
        Isa::Avx512Vnni,
        Isa::AvxVnni,
        Isa::AmxInt8,
        Isa::AmxBf16,
        Isa::Simd128,
    ];

    /// The token as it appears in a report and in `TRACT_CPU_ISA`.
    pub fn name(&self) -> &'static str {
        match self {
            Isa::Neon => "neon",
            Isa::Fp16 => "fp16",
            Isa::DotProd => "dotprod",
            Isa::Sve2 => "sve2",
            Isa::Sme => "sme",
            Isa::Sme2 => "sme2",
            Isa::AppleAmx => "apple-amx",
            Isa::Avx => "avx",
            Isa::Avx2 => "avx2",
            Isa::Fma => "fma",
            Isa::F16c => "f16c",
            Isa::Avx512f => "avx512f",
            Isa::Avx512Vnni => "avx512vnni",
            Isa::AvxVnni => "avxvnni",
            Isa::AmxInt8 => "amx-int8",
            Isa::AmxBf16 => "amx-bf16",
            Isa::Simd128 => "simd128",
        }
    }

    fn from_name(s: &str) -> Option<Isa> {
        Isa::ALL.into_iter().find(|i| i.name() == s)
    }

    /// Where this feature sits in its architecture's ladder, each step meaning "a kernel
    /// written for this needs nothing a kernel written for the step below has, and can do
    /// more". Steps are compared across an architecture's whole kernel pool, so every feature
    /// a kernel can declare has to be placed: an unplaced feature reads as the baseline and
    /// would quietly demote its kernels below every ranked sibling.
    ///
    /// The two architectures share the scale without meeting on it — no host offers features
    /// from both — so `Neon` and `Avx` both sitting at 1 says nothing about each other.
    /// Widening the scale means revisiting [`MAX_TIER`].
    pub fn tier(&self) -> u8 {
        match self {
            // x86: each generation subsumes the last, AMX above the VNNI it needs alongside it.
            Isa::Avx => 1,
            Isa::Avx2 | Isa::Fma | Isa::F16c => 2,
            Isa::Avx512f | Isa::AvxVnni => 3,
            Isa::Avx512Vnni => 4,
            Isa::AmxInt8 | Isa::AmxBf16 => 5,
            // arm: NEON is the armv7 step above bare VFP, and baseline on aarch64 where the
            // ladder continues through the matrix extensions.
            Isa::Neon => 1,
            Isa::Fp16 | Isa::DotProd => 2,
            Isa::Sve2 => 3,
            Isa::Sme | Isa::Sme2 => 4,
            Isa::AppleAmx => 5,
            // wasm has one feature and nothing to outrank.
            Isa::Simd128 => 0,
        }
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The features one machine has.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct IsaSet(u32);

impl IsaSet {
    pub const fn empty() -> IsaSet {
        IsaSet(0)
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
#[derive(Copy, Clone)]
pub struct IsaReq {
    /// Every one of these must be present.
    pub needs: &'static [Isa],
    /// A condition the instruction set cannot express, today only the 512-bit FMA port count.
    /// Whatever hides in here is invisible to enumeration, so it should stay rare.
    pub probe: Option<fn() -> bool>,
}

impl IsaReq {
    /// Runs anywhere its arch does.
    pub const ANY: IsaReq = IsaReq { needs: &[], probe: None };

    pub const fn needing(self, needs: &'static [Isa]) -> IsaReq {
        IsaReq { needs, ..self }
    }

    pub const fn probe(self, probe: fn() -> bool) -> IsaReq {
        IsaReq { probe: Some(probe), ..self }
    }

    pub fn satisfied_by(&self, set: IsaSet) -> bool {
        self.needs.iter().all(|i| set.has(*i)) && self.probe.is_none_or(|p| p())
    }

    /// The most capable lineage step this kernel sits in, feeding the default half of
    /// [`crate::mmm::MatMatMulKer::dynamic_boost`].
    pub fn tier(&self) -> u8 {
        self.needs.iter().map(|i| i.tier()).max().unwrap_or(0)
    }
}

/// What one step up the instruction-set ladder is worth when nothing else is known. A kernel
/// written against a more capable set is assumed better than one written against a less capable
/// one; this is the size of that assumption, in the same units as a declared `boost`.
///
/// It has to stay well under the smallest declared boost, because a declaration is how an
/// exception to the assumption is spelled and it must outweigh every tier it disagrees with:
/// `TIER_BOOST * MAX_TIER` is the widest disagreement a declaration ever has to cross, so
/// keeping it below the smallest declared value in the tree is what makes exceptions win.
pub const TIER_BOOST: isize = 10;

/// The deepest step in the ladder, for the bound above.
pub const MAX_TIER: u8 = 5;

impl fmt::Debug for IsaReq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let needs = self.needs.iter().map(|i| i.name()).collect::<Vec<_>>().join("+");
        if needs.is_empty() {
            f.write_str("any")?;
        } else {
            f.write_str(&needs)?;
        }
        if self.probe.is_some() {
            f.write_str(" +probe")?;
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
    return IsaSet::empty().with(Isa::Simd128);
    #[cfg(not(any(
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    IsaSet::empty()
}

/// `TRACT_CPU_ISA=+sve2,-fp16` edits the probed set, so one knob covers every feature and a
/// cohort this machine is not can be asked what it would dispatch.
fn forced(mut set: IsaSet) -> IsaSet {
    let Some(spec) = crate::knobs::TRACT_CPU_ISA.get() else { return set };
    for token in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let (add, name) = match token.split_at(1) {
            ("+", name) => (true, name),
            ("-", name) => (false, name),
            _ => (true, token),
        };
        match Isa::from_name(name) {
            Some(isa) if add => set = set.with(isa),
            Some(isa) => set = set.without(isa),
            None => log::warn!("TRACT_CPU_ISA: unknown feature {name:?}, ignored"),
        }
    }
    set
}

//! Cross-arch registry of mmm dispatch tiers.
//!
//! A tier is one platform's opinion about which suitable kernel to run, for the accumulators and
//! shapes it claims. Every tier is declared as data, so the whole ladder is enumerable on any
//! host and its precedence is a field rather than the order some `plug` happened to run in.
//!
//! [`preferred`] asks the applicable tiers in descending [`MmmTier::precedence`] and takes the first
//! answer. A tier that returns `None` has no opinion on that query and the next one down is
//! asked, so "only when nothing better answered" needs no condition of its own — the portable
//! rules at precedence 0 are simply the last tier every platform ends on.
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};
use crate::platform::Target;
use tract_data::prelude::DatumType;

/// What a dispatch decision is made for: whose kernels, and what the instruction set offers.
/// `target` is `None` for a platform with no arch tree, which then has only the portable rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Platform {
    pub target: Option<Target>,
    pub isa: IsaSet,
}

impl Platform {
    /// This host: its own arch tree, and the instruction set it probes (or `TRACT_CPU_ISA`
    /// forces).
    pub fn native() -> Platform {
        Platform {
            target: crate::platform::all().map(|s| s.target).find(|t| t.is_native()),
            isa: crate::isa::native(),
        }
    }
}

/// One rung of a platform's dispatch ladder.
pub struct MmmTier {
    /// Arch tree the tier belongs to, `None` for the portable rules every platform ends on.
    pub target: Option<Target>,
    /// Where this tier sits among the tiers of one target: they are asked in descending order and
    /// the first answer wins. It is per-target, and needs only be right between tiers that speak
    /// for the same accumulator — two tiers answering different ones never meet.
    pub precedence: u8,
    /// What to call this rung when reporting the ladder.
    pub name: &'static str,
    /// Whether this tier speaks on `platform` at all: the instruction set it needs, the vendor
    /// or chip it was measured on. Never a shape or an accumulator — those belong to
    /// [`Self::preferred`], which can decline by answering `None`.
    pub applies: fn(&Platform) -> bool,
    /// Which of the suitable kernels this tier would run, `None` for a query it does not claim.
    pub preferred: fn(&Platform, DatumType, &Query, &[Suitable]) -> Option<usize>,
}

inventory::collect!(MmmTier);

/// Every tier this build compiled, whichever platform it speaks for.
pub fn declared() -> impl Iterator<Item = &'static MmmTier> {
    inventory::iter::<MmmTier>()
}

/// The tiers that speak for `platform`, highest precedence first. Ties keep declaration order, which
/// is not stable across builds — two tiers of one target must not share a precedence.
pub fn for_platform(platform: &Platform) -> Vec<&'static MmmTier> {
    let mut tiers: Vec<&'static MmmTier> = declared()
        .filter(|t| t.target.is_none() || t.target == platform.target)
        .filter(|t| (t.applies)(platform))
        .collect();
    tiers.sort_by_key(|t| std::cmp::Reverse(t.precedence));
    log::debug!(
        "mmm tiers for {:?}: {}",
        platform.target,
        tiers.iter().map(|t| t.name).collect::<Vec<_>>().join(" > ")
    );
    tiers
}

/// Which suitable kernel `platform` would run: the answer of the highest-precedence tier that
/// has one. `None` only when no tier claims the query at all.
pub fn preferred(
    platform: &Platform,
    tiers: &[&'static MmmTier],
    accumulator: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<usize> {
    tiers.iter().find_map(|t| (t.preferred)(platform, accumulator, query, suitable))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tiers of one target must not share a precedence: [`for_platform`] sorts by it, and a
    /// tie falls back on declaration order, which is link order and not stable across builds.
    /// Precedence is the whole ordering contract, so a collision is a silent coin toss.
    #[test]
    fn precedence_is_unique_per_target() {
        let tiers: Vec<&MmmTier> = declared().collect();
        for (ix, a) in tiers.iter().enumerate() {
            for b in &tiers[ix + 1..] {
                assert!(
                    a.target != b.target || a.precedence != b.precedence,
                    "tiers {} and {} both claim {:?} precedence {}",
                    a.name,
                    b.name,
                    a.target,
                    a.precedence
                );
            }
        }
    }
}

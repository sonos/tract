//! Cross-arch registry of mmm dispatch tiers.
//!
//! A tier is one architecture's opinion about which suitable kernel to run, for the accumulators
//! and shapes it claims. Every tier is declared as data, so the whole ladder is enumerable on any
//! host and its precedence is a field rather than the order some `plug` happened to run in.
//!
//! [`preferred`] asks the applicable tiers in descending [`MmmTier::precedence`] and takes the
//! first answer. A tier that returns `None` has no opinion on that query and the next one down is
//! asked, so "only when nothing better answered" needs no condition of its own — the portable
//! rules at precedence 0 are simply the last tier every machine ends on.
#[cfg(test)]
use crate::isa::Isa;
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};
use crate::platform::Arch;
use tract_data::prelude::DatumType;

/// One rung of a machine's dispatch ladder.
pub struct MmmTier {
    /// Architecture the tier belongs to, `None` for the portable rules every machine ends on.
    pub arch: Option<Arch>,
    /// Where this tier sits among the tiers of one architecture: they are asked in descending
    /// order and the first answer wins. It needs only be right between tiers that speak for the
    /// same accumulator — two tiers answering different ones never meet.
    pub precedence: u8,
    /// What to call this rung when reporting the ladder.
    pub name: &'static str,
    /// Whether this tier speaks on this machine at all: the instruction set it needs, the vendor
    /// or chip it was measured on. Never a shape or an accumulator — those belong to
    /// [`Self::preferred`], which can decline by answering `None`.
    pub applies: fn(&IsaSet) -> bool,
    /// Which of the suitable kernels this tier would run, `None` for a query it does not claim.
    pub preferred: fn(&IsaSet, DatumType, &Query, &[Suitable]) -> Option<usize>,
}

inventory::collect!(MmmTier);

/// Every tier this build compiled, whichever architecture it speaks for.
pub fn declared() -> impl Iterator<Item = &'static MmmTier> {
    inventory::iter::<MmmTier>()
}

/// The tiers that speak for a machine, highest precedence first. Ties keep declaration order,
/// which is not stable across builds — two tiers of one architecture must not share a precedence.
pub fn for_isa(isa: &IsaSet) -> Vec<&'static MmmTier> {
    let arch = isa.arch();
    let mut tiers: Vec<&'static MmmTier> = declared()
        .filter(|t| t.arch.is_none() || t.arch == arch)
        .filter(|t| (t.applies)(isa))
        .collect();
    tiers.sort_by_key(|t| std::cmp::Reverse(t.precedence));
    log::debug!(
        "mmm tiers for {isa:?}: {}",
        tiers.iter().map(|t| t.name).collect::<Vec<_>>().join(" > ")
    );
    tiers
}

/// Which suitable kernel this machine would run: the answer of the highest-precedence tier that
/// has one. `None` only when no tier claims the query at all.
pub fn preferred(
    isa: &IsaSet,
    tiers: &[&'static MmmTier],
    accumulator: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<usize> {
    tiers.iter().find_map(|t| (t.preferred)(isa, accumulator, query, suitable))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tiers of one architecture must not share a precedence: [`for_isa`] sorts by it, and a
    /// tie falls back on declaration order, which is link order and not stable across builds.
    /// Precedence is the whole ordering contract, so a collision is a silent coin toss.
    #[test]
    fn precedence_is_unique_per_arch() {
        let tiers: Vec<&MmmTier> = declared().collect();
        for (ix, a) in tiers.iter().enumerate() {
            for b in &tiers[ix + 1..] {
                assert!(
                    a.arch != b.arch || a.precedence != b.precedence,
                    "tiers {} and {} both claim {:?} precedence {}",
                    a.name,
                    b.name,
                    a.arch,
                    a.precedence
                );
            }
        }
    }

    /// Every feature belongs to exactly one architecture, so a tier's own architecture and the
    /// features its `applies` asks for cannot disagree — a tier that required another
    /// architecture's feature would simply never fire.
    #[test]
    fn a_tier_only_needs_its_own_architecture() {
        for tier in declared() {
            let Some(arch) = tier.arch else { continue };
            for isa in Isa::ALL.into_iter().filter(|i| !i.is_arch()) {
                if (tier.applies)(&IsaSet::of_arch(arch).with(isa))
                    != (tier.applies)(&IsaSet::of_arch(arch))
                {
                    assert_eq!(
                        isa.arch(),
                        arch,
                        "tier {} is {arch:?} but reacts to {isa}, which is {:?}",
                        tier.name,
                        isa.arch()
                    );
                }
            }
        }
    }
}

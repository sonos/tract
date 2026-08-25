//! Cross-arch registry of mmm dispatch tiers.
//!
//! A tier is one architecture's opinion about which suitable kernel to run, for the accumulators
//! and shapes it claims. Every tier is declared as data, so the whole ladder is enumerable on any
//! host and its precedence is a field rather than an order of registration.
//!
//! [`preferred`] asks the applicable tiers in descending [`MmmTier::precedence`] and takes the
//! first answer that the query can actually reach. A tier with no opinion, or one naming a kernel
//! this query has no suitable entry for, leaves it to the next tier down — so "only when nothing
//! better answered" needs no condition of its own, and the generic rules at precedence 0 are
//! simply the last tier every machine ends on.
use crate::isa::Arch;
#[cfg(test)]
use crate::isa::Isa;
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};
use tract_data::prelude::DatumType;

/// One rung of a machine's dispatch ladder.
pub struct MmmTier {
    /// Architecture the tier belongs to, `None` for the generic rules every machine ends on.
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
    /// Which kernel this tier would run, by name, `None` for a query it does not claim. A tier
    /// answers with the kernel it wants and [`preferred`] holds that answer to the suitable list,
    /// so naming one the query cannot reach is the same as having no opinion: the next tier down
    /// is asked. Only a tier picking *from* the list — a cost model weighing candidates — needs
    /// to read `suitable` at all.
    pub preferred: fn(&IsaSet, DatumType, &Query, &[Suitable]) -> Option<&'static str>,
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
    tiers.iter().find_map(|t| {
        let name = (t.preferred)(isa, accumulator, query, suitable)?;
        crate::mmm::suitable_named(suitable, name)
    })
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
    /// A tier that names a kernel must somewhere name one the query can reach. Naming an
    /// unreachable kernel is how a tier defers to the one below, so a tier doing it at every step
    /// and for every accumulator is dead dispatch that no pick can expose: the machine falls
    /// through and still runs something. A tier gated on a knob that is off never answers here at
    /// all, and is not held to this; setting the knob brings it in.
    ///
    /// Only the steps this host can execute, unlike the questions asked of the routine registry:
    /// the mmm pool holds a compiled kernel only where the CPU can run it, so a step above this
    /// machine has an empty pool and nothing there is reachable by construction.
    #[test]
    fn a_tier_that_answers_names_a_reachable_kernel() {
        let mut answered = std::collections::HashSet::new();
        let mut reached = std::collections::HashSet::new();
        let native = crate::isa::native();
        for isa in IsaSet::every_ladder().filter(|l| l.iter().all(|i| native.has(i))) {
            let dispatch = crate::MmmDispatch::for_isa(isa);
            for acc in [DatumType::F32, DatumType::F16, DatumType::I32] {
                let query = Query::plain(acc, None, None, None);
                let suitable = dispatch.suitable(&query);
                for tier in dispatch.tiers() {
                    let Some(name) = (tier.preferred)(&isa, acc, &query, &suitable) else {
                        continue;
                    };
                    answered.insert(tier.name);
                    if crate::mmm::suitable_named(&suitable, name).is_some() {
                        reached.insert(tier.name);
                    }
                }
            }
        }
        let unheard: Vec<&str> =
            answered.iter().copied().filter(|name| !reached.contains(name)).collect();
        assert!(unheard.is_empty(), "these tiers name a kernel no machine can reach: {unheard:?}");
    }
}

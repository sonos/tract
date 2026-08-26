//! What a caller needs a matmul for, and what it can be given in return.
//!
//! Selection narrows one set of kernels, and each rung is named after the property it tests:
//! *declared* is every kernel the inventory knows ([`crate::mmm_routines::declared`]), *built*
//! those whose body this build compiled ([`MatMatMul::built`]), *runnable* those the host can
//! also execute ([`MatMatMul::runnable`], collected as [`crate::MmmDispatch::runnable`]), *suitable*
//! those that can answer a given [`Query`] ([`crate::MmmDispatch::suitable`]).
//!
//! The last step is not a filter but a choice, so it has no set of its own: *preference* is the
//! ordering a platform imposes over the suitable ones ([`crate::MmmDispatch::preferred`]), and it is a
//! property of the platform, never of a kernel. A [`Suitable`] is one way to compute
//! the matmul that fits the query — kernel, packing, extractor.

use dyn_eq::DynEq;
use tract_data::itertools::Itertools;
use tract_data::prelude::{DatumType, TVec, tvec};

use super::{MMMInputFormat, MatMatMul, PanelExtractor};
use crate::WeightType;

/// One way to compute a matmul: a kernel, which of its packings to use, and the panel
/// extractor to reach that packing when the weights are not already in it.
pub type Suitable = (Box<dyn MatMatMul>, usize, Option<PanelExtractor>);

/// A matmul as kernel selection sees it: the operand types a kernel must handle, and the
/// shape where the caller knows it. A `None` dim is one the caller cannot pin — a streaming
/// axis, or an `n` still symbolic at optimisation time.
#[derive(Clone)]
pub struct Query {
    pub weight: WeightType,
    pub activation: DatumType,
    /// Internal accumulator types the caller accepts.
    pub accumulators: TVec<DatumType>,
    /// The datum type the kernel must be able to store, when the caller constrains it.
    pub store: Option<DatumType>,
    /// Whether a kernel reached through a panel extractor is acceptable. It is not for a
    /// caller that packs its weights once, ahead of time: the extractor would then run on
    /// every panel of every call.
    pub allow_extractor: bool,
    pub m: Option<usize>,
    pub k: Option<usize>,
    pub n: Option<usize>,
}

impl Query {
    /// The plain matmul a platform policy reasons about: operands in the accumulator's own type,
    /// `i8` operands for an integer accumulator, no store constraint and extractors allowed.
    /// Mixed precision — f16 operands accumulated in f32 — is not one of these, so a policy
    /// never speaks for it.
    pub fn plain(
        accumulator: DatumType,
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Query {
        let operand = if accumulator == DatumType::I32 { DatumType::I8 } else { accumulator };
        Query {
            weight: operand.into(),
            activation: operand,
            accumulators: tvec!(accumulator),
            store: None,
            allow_extractor: true,
            m,
            k,
            n,
        }
    }
}

/// Where the kernel called `name` sits in `suitable`, or `None` when it is not suitable for this
/// query. This is how a policy tier answers: naming a kernel the query never reached defers to
/// the tier below instead of going unheard.
pub fn suitable_named(suitable: &[Suitable], name: &str) -> Option<usize> {
    suitable.iter().position(|(mmm, _, _)| mmm.name() == name)
}

/// Keep only the suitable kernels nothing supersedes: a kernel written for this architecture
/// beats portable Rust whatever their instruction sets, and among kernels of one kind the
/// instruction-set level and the declared boost decide. Everything tied at the top is kept —
/// the shape rules run over what is left. First rung of the generic policy, applied before any
/// shape reasoning.
pub fn retain_best(suitable: &mut Vec<Suitable>) {
    fn key(mmm: &dyn MatMatMul) -> (bool, isize) {
        (mmm.arch().is_some(), mmm.preference())
    }
    if let Some(best) = suitable.iter().map(|(mmm, _, _)| key(&**mmm)).max() {
        suitable.retain(|(mmm, _, _)| key(&**mmm) == best);
    }
}

/// The generic policy’s shape rules: which suitable kernel to use once the query pins `n`, and
/// `None` when it does not — a caller facing a symbolic `n` has to decide for itself. Ties
/// go to the last one, so the order of the suitable list breaks them.
pub fn pick_by_shape(query: &Query, suitable: &[Suitable]) -> Option<usize> {
    match query.n {
        // A true GEMV first, then no extractor to pay for, then the widest tile.
        Some(1) => suitable
            .iter()
            .enumerate()
            .max_by_key(|(_, (mmm, _, pe))| (mmm.nr() == 1, pe.is_none(), mmm.mr()))
            .map(|(ix, _)| ix),
        // For a 2D matmul a GEMV kernel (nr == 1) is a poor fit: it processes one output
        // column per pass. Demote it so it never wins the `nr * mr` tie against a square
        // tile (i8 64x1 and 8x8 both have nr * mr == 64). Ordering among nr > 1 kernels is
        // left untouched.
        Some(n) if n > 1 => suitable
            .iter()
            .enumerate()
            .max_by_key(|(_, (mmm, _, pe))| (pe.is_none(), mmm.nr() > 1, mmm.nr() * mmm.mr()))
            .map(|(ix, _)| ix),
        _ => None,
    }
}

/// Everything this platform brings to a matrix multiplication: the kernels it can run, the tiers
/// that rank them, and the panel extractors that reach a kernel's packing from a foreign one.
/// Nothing else lives here -- the single-winner kernels are [`routines`].
pub struct MmmDispatch {
    /// The architecture these kernels are for and what its instruction set offers. Everything mmm
    /// selection is a function of it: the runnable set, and which tiers speak.
    isa: crate::isa::IsaSet,
    /// The applicable tiers, highest precedence first, resolved once from [`Self::isa`].
    tiers: Vec<&'static crate::mmm_tiers::MmmTier>,
    runnable: Vec<Box<dyn MatMatMul>>,
    panel_extractors: Vec<PanelExtractor>,
}

impl MmmDispatch {
    /// What a platform runs: its runnable kernels, the tiers that speak for it, and the panel
    /// extractors it can reach. All three are functions of the instruction set, so this is the
    /// whole of it -- nothing is installed by hand afterwards.
    pub fn for_isa(isa: crate::isa::IsaSet) -> MmmDispatch {
        let mut dispatch = MmmDispatch {
            isa,
            tiers: crate::mmm_tiers::for_isa(&isa),
            runnable: vec![],
            panel_extractors: crate::mmm_routines::extractors_for(&isa),
        };
        dispatch.runnable = match isa.arch() {
            Some(target) => crate::mmm_routines::runnable_for(target),
            None => crate::mmm_routines::runnable(),
        };
        dispatch
    }

    /// What this host runs, resolved once. The runnable set is every kernel this build compiled
    /// that the CPU can execute, so it is worth keeping rather than rebuilding per query.
    pub fn native() -> &'static MmmDispatch {
        lazy_static::lazy_static! {
            static ref NATIVE: MmmDispatch = MmmDispatch::for_isa(crate::isa::native());
        }
        &NATIVE
    }

    /// Every kernel this host can execute: built into this build, and declaring an instruction
    /// set the CPU has. What selection narrows down from.
    pub fn runnable(&self) -> &[Box<dyn MatMatMul>] {
        &self.runnable
    }

    pub fn all_possible_packing(
        &self,
        weight_type: impl Into<crate::WeightType>,
    ) -> impl Iterator<Item = &dyn MMMInputFormat> {
        let weight_type = weight_type.into();
        self.runnable
            .iter()
            .flat_map(|m| m.packings())
            .map(|p| &*p.0)
            .flat_map(move |p| {
                let mut packs: Vec<&dyn MMMInputFormat> = vec![];
                if p.precursor() == weight_type {
                    packs.push(p)
                };
                for pe in &self.panel_extractors {
                    if pe.from.precursor() == weight_type && pe.to.dyn_eq(p) {
                        packs.push(&*pe.from);
                    }
                }
                packs.into_iter()
            })
            .sorted_by_key(|p| p.to_string())
            .dedup()
    }

    /// Every way this build can compute the queried matmul: a kernel, which of its packings
    /// to use, and the panel extractor to reach that packing when the weights are not
    /// already in it. The query’s dims are not consulted — a kernel is suitable or not
    /// whatever the shape.
    ///
    /// The one enumeration behind both matmul lowerings — how one is *chosen* from
    /// the list is the caller's business.
    pub fn suitable(&self, query: &Query) -> Vec<Suitable> {
        self.runnable
            .iter()
            .filter(|mmm| {
                query.accumulators.contains(&mmm.internal_type())
                    && query.store.is_none_or(|s| mmm.stores().contains(&s))
            })
            .flat_map(|mmm| mmm.packings().iter().enumerate().map(move |(ix, p)| (mmm, ix, p)))
            .filter(|(_, _, (_, b))| {
                b.precursor().as_dt().is_some_and(|dt| dt == query.activation.unquantized())
            })
            .filter_map(|(mmm, ix, (a, _))| {
                if a.precursor() == query.weight {
                    Some((mmm.clone(), ix, None))
                } else if query.allow_extractor {
                    self.panel_extractors
                        .iter()
                        .find(|pe| pe.from.precursor() == query.weight && pe.to.dyn_eq(&**a))
                        .map(|pe| (mmm.clone(), ix, Some(pe.clone())))
                } else {
                    None
                }
            })
            .collect()
    }

    /// This platform’s tiers, highest precedence first: the ladder [`Self::preferred`] walks.
    pub fn tiers(&self) -> &[&'static crate::mmm_tiers::MmmTier] {
        &self.tiers
    }

    /// The platform's choice among `suitable`, or `None` when no tier has an opinion — none
    /// claimed this accumulator, or the query is not the plain matmul the tiers reason about.
    /// A generic answer counts as no opinion: the precedence-0 tier answers for every platform, so
    /// getting one of its kernels back means no arch tier claimed the query.
    pub fn preferred(&self, query: &Query, suitable: &[Suitable]) -> Option<Suitable> {
        let crate::WeightType::Plain(weight) = &query.weight else { return None };
        if weight.unquantized() != query.activation.unquantized() {
            return None;
        }
        let acc = *query.accumulators.first()?;
        let ix = crate::mmm_tiers::preferred(&self.isa, &self.tiers, acc, query, suitable)?;
        let chosen = &suitable[ix];
        chosen.0.arch().is_some().then(|| chosen.clone())
    }

    /// One kernel for the query, for a caller that needs an answer now: the platform policy's
    /// pick where it has an opinion, then the generic rules, then the widest extractor-free
    /// tile. That last resort is what a caller with no fallback of its own needs when `n` is
    /// unknown or degenerate — a caller that can do better with the whole list, as einsum can
    /// for a symbolic `n`, should walk the suitable kernels itself. `None` only when nothing suitable
    /// exists at all.
    pub fn pick(&self, query: &Query) -> Option<Suitable> {
        let mut suitable = self.suitable(query);
        if let Some(chosen) = self.preferred(query, &suitable) {
            return Some(chosen);
        }
        retain_best(&mut suitable);
        if suitable.len() == 1 {
            return Some(suitable.remove(0));
        }
        if let Some(ix) = pick_by_shape(query, &suitable) {
            return Some(suitable.swap_remove(ix));
        }
        let ix = suitable
            .iter()
            .enumerate()
            .max_by_key(|(_, (mmm, _, pe))| (pe.is_none(), mmm.nr() > 1, mmm.nr() * mmm.mr()))
            .map(|(ix, _)| ix)?;
        Some(suitable.swap_remove(ix))
    }

    pub fn panel_extractors(&self) -> &[PanelExtractor] {
        &self.panel_extractors
    }

    /// The kernel this platform would run for a plain matmul of these dims, for a caller
    /// introspecting dispatch rather than performing it. Unlike [`MmmDispatch::preferred`] it reports
    /// the tiers' answer whatever kind of kernel it is, and it never falls back on the generic rules:
    /// `None` means no tier had anything to say.
    pub fn preferred_kernel(
        &self,
        accumulator: DatumType,
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Option<Box<dyn MatMatMul>> {
        let query = Query::plain(accumulator, m, k, n);
        let suitable = self.suitable(&query);
        let ix =
            crate::mmm_tiers::preferred(&self.isa, &self.tiers, accumulator, &query, &suitable)?;
        Some(suitable[ix].0.clone())
    }
}

impl crate::isa::Arch {
    /// Dispatch as `arch` sees it: its kernels, from [`crate::mmm_routines::runnable_for`], under its own
    /// tiers. Answers which kernel that architecture would choose for a shape, from any host. What it
    /// cannot reproduce is a hardware probe, so for a foreign arch it starts from the plain
    /// architecture and nothing else: a cohort behind a feature is reached by naming that feature in
    /// `TRACT_CPU_ISA`, which is checked against this architecture rather than the host's. `None` when
    /// the architecture's tree was not compiled in; see the `foreign-inventory` feature.
    pub fn inspect(self) -> Option<MmmDispatch> {
        if !crate::mmm_routines::declared().any(|r| (r.make)().arch() == Some(self)) {
            return None;
        }
        let isa = if self.is_native() {
            crate::isa::native()
        } else {
            crate::isa::forced(crate::isa::IsaSet::of_arch(self))
        };
        Some(MmmDispatch::for_isa(isa))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmm::MMMInputFormat;
    use tract_data::prelude::{Datum, f16};

    /// The accumulators a platform policy speaks for, each standing for the plain query
    /// [`Query::plain`] builds from it.
    fn accumulators() -> Vec<DatumType> {
        vec![f32::datum_type(), f16::datum_type(), i32::datum_type()]
    }

    /// A caller facing a symbolic `n` picks a packing group and needs both roles out of it: a
    /// matvec kernel for the pulses where `n` is 1 and a matrix kernel for the rest. A preference
    /// that drops a group's matvec while keeping its matrix kernel leaves that caller a group it
    /// cannot use for both, and it silently falls back on a narrower group instead.
    ///
    /// Only the boost is held to this. The architecture key dropping the matvec is it working as
    /// intended — portable Rust has no business being preferred beside a kernel written for
    /// this machine, and a group whose only matvec is generic is a group the caller is right to
    /// treat as matrix-only.
    #[test]
    fn the_preference_keeps_a_kept_group_usable() {
        let dispatch = crate::MmmDispatch::native();
        for acc in accumulators() {
            let query = Query::plain(acc, None, None, None);
            let all = dispatch.suitable(&query);
            let Some(best) = all.iter().map(|(mmm, _, _)| mmm.arch().is_some()).max() else {
                continue;
            };
            let peers: Vec<Suitable> =
                all.iter().filter(|(mmm, _, _)| mmm.arch().is_some() == best).cloned().collect();
            let mut kept = peers.clone();
            retain_best(&mut kept);
            let kept: Vec<&str> = kept.iter().map(|(mmm, _, _)| mmm.name()).collect();
            for group in packing_groups(&peers) {
                let matvecs: Vec<&str> =
                    group.iter().filter(|c| c.0.nr() == 1).map(|c| c.0.name()).collect();
                if matvecs.is_empty() || !group.iter().any(|c| kept.contains(&c.0.name())) {
                    continue;
                }
                assert!(
                    matvecs.iter().any(|name| kept.contains(name)),
                    "preference kept {:?} of the {acc:?} packing group {:?} but dropped its \
                     matvec kernels {matvecs:?}",
                    group
                        .iter()
                        .filter(|c| kept.contains(&c.0.name()))
                        .map(|c| c.0.name())
                        .collect::<Vec<_>>(),
                    group[0].0.packings()[group[0].1].0
                );
            }
        }
    }

    /// The suitable kernels a caller with a symbolic `n` has to choose between, grouped by the weight
    /// packing it would have to commit to, as `kernel_selection::strategize` groups them.
    fn packing_groups(suitable: &[Suitable]) -> Vec<Vec<&Suitable>> {
        let mut groups: Vec<(&dyn MMMInputFormat, Vec<&Suitable>)> = vec![];
        'entry: for entry in suitable {
            let (mmm, packing, extractor) = entry;
            let left: &dyn MMMInputFormat =
                extractor.as_ref().map(|pe| &*pe.from).unwrap_or(&*mmm.packings()[*packing].0);
            for group in &mut groups {
                if let Some(merged) = group.0.merge_with(left) {
                    group.0 = merged;
                    group.1.push(entry);
                    continue 'entry;
                }
            }
            groups.push((left, vec![entry]));
        }
        groups.into_iter().map(|(_, group)| group).collect()
    }
}

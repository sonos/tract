//! What a caller needs a matmul for, and what it can be given in return.

use tract_data::prelude::{DatumType, TVec};

use super::{MatMatMul, PanelExtractor};
use crate::WeightType;

/// One way to compute a matmul: a kernel, which of its packings to use, and the panel
/// extractor to reach that packing when the weights are not already in it.
pub type Candidate = (Box<dyn MatMatMul>, usize, Option<PanelExtractor>);

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

/// Keep only the candidates in the best implementation tier: quality dominates, and a
/// kernel's dynamic boost breaks ties within a tier. First rung of the portable policy,
/// applied before any shape reasoning.
pub fn retain_best_quality(candidates: &mut Vec<Candidate>) {
    fn score(mmm: &dyn MatMatMul) -> isize {
        -(mmm.quality().cost() as isize * 1000) + mmm.dynamic_boost()
    }
    if let Some(best) = candidates.iter().map(|(mmm, _, _)| score(&**mmm)).max() {
        candidates.retain(|(mmm, _, _)| score(&**mmm) == best);
    }
}

/// The portable policy's shape rules: which candidate to use once the query pins `n`, and
/// `None` when it does not — a caller facing a symbolic `n` has to decide for itself. Ties
/// go to the last candidate, so pool order breaks them.
pub fn pick_by_shape(query: &Query, candidates: &[Candidate]) -> Option<usize> {
    match query.n {
        // A true GEMV first, then no extractor to pay for, then the widest tile.
        Some(1) => candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, (mmm, _, pe))| (mmm.nr() == 1, pe.is_none(), mmm.mr()))
            .map(|(ix, _)| ix),
        // For a 2D matmul a GEMV kernel (nr == 1) is a poor fit: it processes one output
        // column per pass. Demote it so it never wins the `nr * mr` tie against a square
        // tile (i8 64x1 and 8x8 both have nr * mr == 64). Ordering among nr > 1 kernels is
        // left untouched.
        Some(n) if n > 1 => candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, (mmm, _, pe))| (pe.is_none(), mmm.nr() > 1, mmm.nr() * mmm.mr()))
            .map(|(ix, _)| ix),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmm::{ImplementationQuality, MMMInputFormat};
    use tract_data::prelude::{Datum, f16, tvec};

    /// The plain queries a platform policy reasons about, as (weight and activation,
    /// accumulator). A mixed-precision query -- f16 operands accumulated in f32 -- is not one of
    /// them: the policy is only handed the accumulator, so it cannot name a kernel that packs
    /// f16, and [`crate::Ops::rank`] takes its unreachable answer as no opinion.
    fn operands() -> Vec<(DatumType, DatumType)> {
        vec![
            (f32::datum_type(), f32::datum_type()),
            (f16::datum_type(), f16::datum_type()),
            (i8::datum_type(), i32::datum_type()),
        ]
    }

    /// Shapes spanning the tile sizes in the pools, plus the `None` a symbolic dim leaves.
    const DIMS: &[Option<usize>] =
        &[None, Some(1), Some(2), Some(4), Some(8), Some(15), Some(64), Some(512)];

    fn query(
        operands: (DatumType, DatumType),
        m: Option<usize>,
        k: Option<usize>,
        n: Option<usize>,
    ) -> Query {
        Query {
            weight: operands.0.into(),
            activation: operands.0,
            accumulators: tvec!(operands.1),
            store: None,
            allow_extractor: true,
            m,
            k,
            n,
        }
    }

    /// [`crate::Ops::rank`] honours the platform policy by finding its answer in the candidate
    /// list, and silently falls back on the portable rules when it is not there. A policy that
    /// names a kernel the enumeration does not offer for that query is therefore inert, and
    /// nothing else says so.
    #[test]
    fn the_policy_only_names_kernels_the_pool_offers() {
        let ops = crate::ops();
        for operands in operands() {
            for m in DIMS {
                for k in DIMS {
                    for n in DIMS {
                        let query = query(operands, *m, *k, *n);
                        let Some(named) = (ops.mmm_policy())(operands.1, *m, *k, *n) else {
                            continue;
                        };
                        if named.quality() != ImplementationQuality::ManuallyOptimized {
                            continue;
                        }
                        let candidates = ops.candidates(&query);
                        assert!(
                            candidates.iter().any(|(mmm, _, _)| mmm.name() == named.name()),
                            "policy names {} for {operands:?} m={m:?} k={k:?} n={n:?}, \
                             which the enumeration does not offer: {:?}",
                            named.name(),
                            candidates.iter().map(|(mmm, _, _)| mmm.name()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }

    /// A caller facing a symbolic `n` picks a packing group and needs both roles out of it: a
    /// matvec kernel for the pulses where `n` is 1 and a matrix kernel for the rest. Ranking a
    /// group's matvec away while keeping its matrix kernel leaves that caller a group it cannot
    /// use for both, and it silently falls back on a narrower group instead.
    #[test]
    fn the_ranking_keeps_a_kept_group_usable() {
        let ops = crate::ops();
        for operands in operands() {
            let query = query(operands, None, None, None);
            let all = ops.candidates(&query);
            let mut kept = all.clone();
            retain_best_quality(&mut kept);
            let kept: Vec<&str> = kept.iter().map(|(mmm, _, _)| mmm.name()).collect();
            for group in packing_groups(&all) {
                let matvecs: Vec<&str> =
                    group.iter().filter(|c| c.0.nr() == 1).map(|c| c.0.name()).collect();
                if matvecs.is_empty() || !group.iter().any(|c| kept.contains(&c.0.name())) {
                    continue;
                }
                assert!(
                    matvecs.iter().any(|name| kept.contains(name)),
                    "ranking kept {:?} of the {operands:?} packing group {:?} but dropped its \
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

    /// The candidates a caller with a symbolic `n` has to choose between, grouped by the weight
    /// packing it would have to commit to, as `kernel_selection::strategize` groups them.
    fn packing_groups(candidates: &[Candidate]) -> Vec<Vec<&Candidate>> {
        let mut groups: Vec<(&dyn MMMInputFormat, Vec<&Candidate>)> = vec![];
        'candidate: for candidate in candidates {
            let (mmm, packing, extractor) = candidate;
            let left: &dyn MMMInputFormat =
                extractor.as_ref().map(|pe| &*pe.from).unwrap_or(&*mmm.packings()[*packing].0);
            for group in &mut groups {
                if let Some(merged) = group.0.merge_with(left) {
                    group.0 = merged;
                    group.1.push(candidate);
                    continue 'candidate;
                }
            }
            groups.push((left, vec![candidate]));
        }
        groups.into_iter().map(|(_, group)| group).collect()
    }
}

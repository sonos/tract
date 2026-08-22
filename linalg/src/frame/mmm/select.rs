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

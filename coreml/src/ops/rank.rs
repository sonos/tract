//! Shared rank-shape helpers used by multiple per-op translators.
//!
//! In-MLPackage convention: every tensor is rank ≥ 4 (with leading `1`s)
//! so chains stay rank-4 internally. The `*_external_shape` boundary
//! mechanism (see general_matmul.rs / reshape.rs) handles the rare
//! rank > 5 case via leading-unit-dim stripping.

/// Pad a tract-view shape to `target_rank` by prepending `1`s.
/// Used to normalise CHW (rank 3) to NCHW (rank 4) at the MLPackage
/// boundary, matching the in-package rank-4-padding convention.
pub fn pad_to_rank(shape: &[i64], target_rank: usize) -> Vec<i64> {
    let pad = target_rank.saturating_sub(shape.len());
    if pad == 0 {
        return shape.to_vec();
    }
    std::iter::repeat_n(1i64, pad).chain(shape.iter().copied()).collect()
}

/// Convenience: pad to rank 4 (the in-MLPackage convention).
pub fn pad_to_rank_4(shape: &[i64]) -> Vec<i64> {
    pad_to_rank(shape, 4)
}

/// Strip leading unit dims (`1`s) from a shape until rank ≤ 5.
/// Returns `Some(stripped)` if successful, `None` if rank > 5 has no
/// strippable leading unit dims. Used at the MLPackage I/O boundary
/// to bridge tract's higher-rank views down to MIL's rank-5 cap.
pub fn try_strip_to_rank5(shape: &[i64]) -> Option<Vec<i64>> {
    if shape.len() <= 5 {
        return Some(shape.to_vec());
    }
    let need = shape.len() - 5;
    if shape.iter().take(need).all(|&d| d == 1) { Some(shape[need..].to_vec()) } else { None }
}
